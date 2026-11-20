#!/usr/bin/env python
"""Convert RL Token and actor .pt checkpoints to a LeRobot policy directory.

Provide an actor-critic checkpoint and a demo adaptation checkpoint. If the
actor-critic checkpoint includes its own RL Token, that encoder takes priority.
The output contains the RL Token encoder and actor, excluding the VLA backbone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evo_rlt.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt import RLTPretrainedPolicy


def convert(
    rl_token_ckpt: str | None,
    ac_ckpt: str,
    output_dir: str,
    config_overrides: dict | None = None,
) -> Path:
    """Load the deployment policy from .pt and save a self-contained RL head."""
    if not ac_ckpt:
        raise ValueError("An actor-critic checkpoint is required for conversion.")
    actor_checkpoint = torch.load(ac_ckpt, map_location="cpu", weights_only=True)
    if "rl_token_state_dict" in actor_checkpoint:
        rl_token_checkpoint = actor_checkpoint
    elif rl_token_ckpt:
        rl_token_checkpoint = torch.load(rl_token_ckpt, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Provide --rl-token-ckpt when the actor-critic checkpoint has no RL Token.")

    overrides = dict(config_overrides or {})
    metadata = actor_checkpoint.get("metadata") or {}
    for name in ("action_dim", "proprio_dim", "chunk_length"):
        if name in metadata:
            overrides.setdefault(name, metadata[name])
    overrides.update(rl_token_ckpt_path="", ac_ckpt_path="")
    config = RLTPretrainedConfig(**overrides)
    policy = RLTPretrainedPolicy(
        config, rl_token_checkpoint=rl_token_checkpoint, ac_checkpoint=actor_checkpoint,
    )
    out = Path(output_dir)
    policy.save_pretrained(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-token-ckpt", type=str, help="Path to demo_adapt_checkpoint.pt")
    parser.add_argument("--ac-ckpt", type=str, required=True, help="Path to the actor-critic checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for model.safetensors and config.json")
    parser.add_argument("--vla-path", type=str, default="lerobot/pi05_base",
                        help="VLA pretrained path to store in config")
    parser.add_argument("--task", type=str, default="", help="Task instruction to store in config")
    args = parser.parse_args()

    overrides = {
        "vla_pretrained_path": args.vla_path,
        "task_instruction": args.task,
    }
    out = convert(args.rl_token_ckpt, args.ac_ckpt, args.output_dir, config_overrides=overrides)
    print(f"Saved converted policy to {out}")


if __name__ == "__main__":
    main()
