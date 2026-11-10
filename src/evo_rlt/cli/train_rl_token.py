#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.cli.common import build_pi05_policy, configure_logging, load_training_config

logger = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RL token module on a raw demo dataset.")
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--demo-dataset-path", default=None)
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config.")
    parser.add_argument("--output-dir", default="outputs/rl_token")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vla-ft-weight", type=float, default=0.0)
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default=None, help="Path to an RL token checkpoint to resume from.")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--token-pool-size", type=int, default=0, help="Pool prefix tokens before RL token encoding (0 disables pooling).")
    parser.add_argument("--image-only", action="store_true", help="Drop language tokens before RL token encode (image-patch tokens only).")
    parser.add_argument("--active-cameras", default=None,
                        help="Comma-separated camera names (e.g. 'right_wrist' or 'left_wrist,right_wrist'). Implies image-only and overrides it.")
    parser.add_argument("--norm-stats", default=None, help="Path to a .pt file with 'std' tensor for per-dim weighted MSE.")
    parser.add_argument("--norm-gamma", type=float, default=0.0, help="Per-dim weighting exponent. 0=raw MSE, 0.5=partial whitening, 1=full whitening.")
    parser.add_argument("--num-rl-tokens", type=int, default=None, help="Override config.rl_token.num_rl_tokens.")
    parser.add_argument("--enc-layers", type=int, default=None, help="Override config.rl_token.enc_layers.")
    parser.add_argument("--dec-layers", type=int, default=None, help="Override config.rl_token.dec_layers.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="VLA model dtype")
    parser.add_argument("--vla-cache-dir", default=None, help="Optional Pi0.5 cache directory.")
    parser.add_argument("--tokenizer-path", default=None, help="PaliGemma tokenizer repo id or local snapshot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from evo_rlt.core.algorithm import RLTAlgorithm
    from evo_rlt.adapters.lerobot.demo_loader import make_demo_loader
    from evo_rlt.core.trainer import demo_adaptation

    config = load_training_config(args.config)
    if args.steps is not None:
        config.demo_adaptation.steps = args.steps
    if args.batch_size is not None:
        config.demo_adaptation.batch_size = args.batch_size
    if args.lr is not None:
        config.demo_adaptation.lr = args.lr
    config.demo_adaptation.vla_ft_weight = args.vla_ft_weight
    if args.num_rl_tokens is not None:
        config.rl_token.num_rl_tokens = args.num_rl_tokens
    if args.enc_layers is not None:
        config.rl_token.enc_layers = args.enc_layers
    if args.dec_layers is not None:
        config.rl_token.dec_layers = args.dec_layers

    active_cameras = args.active_cameras.split(",") if args.active_cameras else None

    logger.info("Loading pi0.5 from %s", args.model_path)
    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        vla_cache_dir=args.vla_cache_dir,
        image_only=args.image_only,
        active_cameras=active_cameras,
        tokenizer_path=args.tokenizer_path,
    )
    algorithm = RLTAlgorithm(policy, config)
    logger.info(
        "Policy ready. RL token params: %.1fM",
        sum(parameter.numel() for parameter in policy.rl_token.parameters()) / 1e6,
    )

    rl_token_full = algorithm.build_rl_token_full(args.device)
    start_step = 0
    prior_losses = None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)
        rl_token_full.load_state_dict(checkpoint["rl_token_state_dict"], strict=False)
        start_step = checkpoint.get("step", 0)
        prior_losses = checkpoint.get("losses", [])
        logger.info("Resumed RL token checkpoint at step %d", start_step)

    dim_std = None
    if args.norm_stats:
        stats = torch.load(args.norm_stats, map_location=args.device, weights_only=False)
        dim_std = stats["std"].to(args.device)
        logger.info("Loaded dim_std from %s (shape=%s, gamma=%.2f)", args.norm_stats, tuple(dim_std.shape), args.norm_gamma)

    demo_loader = make_demo_loader(
        dataset_path=args.demo_dataset_path,
        batch_size=config.demo_adaptation.batch_size,
        chunk_length=config.vla_horizon,
        num_workers=args.num_workers,
        device=args.device,
    )
    trainable_parameters = list(rl_token_full.parameters())
    if args.vla_ft_weight > 0:
        trainable_parameters.extend(policy.vla.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.demo_adaptation.lr)

    logger.info(
        "Training RL token: steps=%d batch_size=%d lr=%.2e",
        config.demo_adaptation.steps,
        config.demo_adaptation.batch_size,
        config.demo_adaptation.lr,
    )
    start_time = time.time()
    losses = demo_adaptation(
        algorithm=algorithm,
        config=config,
        demo_loader=demo_loader,
        demo_optimizer=optimizer,
        rl_token_full=rl_token_full,
        save_dir=args.output_dir,
        save_every=args.save_every,
        start_step=start_step,
        prior_losses=prior_losses,
        metadata={
            "vla_model": args.model_path,
            "dataset": args.demo_dataset_path,
            "image_only": args.image_only,
            "active_cameras": active_cameras,
            "num_rl_tokens": config.rl_token.num_rl_tokens,
            "normalize_encoder_output": config.rl_token.normalize_encoder_output,
            "norm_gamma": args.norm_gamma,
            "norm_stats": args.norm_stats,
        },
        dim_std=dim_std,
        norm_gamma=args.norm_gamma,
    )
    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else 0.0
    avg_last_100 = sum(losses[-100:]) / min(len(losses), 100) if losses else 0.0
    logger.info("RL token training finished in %.1fs. final=%.4f avg100=%.4f", elapsed, final_loss, avg_last_100)


if __name__ == "__main__":
    main()
