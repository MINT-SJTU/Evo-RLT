from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from torch import nn

RL_TOKEN_DIR = "rl_token"
RL_TOKEN_STATE_FILENAME = "state_dict.safetensors"
RL_TOKEN_META_FILENAME = "meta.json"
RL_TOKEN_META_FORMAT_VERSION = 1


def save_rl_token_state(
    checkpoint_dir: Path,
    rl_token: nn.Module,
    rl_token_cfg: Any,
    num_image_tokens: int | None = None,
) -> None:
    rl_token_dir = checkpoint_dir / RL_TOKEN_DIR
    rl_token_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.detach().contiguous().cpu() for k, v in rl_token.state_dict().items()}
    save_file(state_dict, rl_token_dir / RL_TOKEN_STATE_FILENAME)

    module_config = {
        "token_dim": int(getattr(rl_token, "token_dim")),
        "nhead": int(rl_token_cfg.nhead),
        "num_enc_layers": int(rl_token_cfg.num_enc_layers),
        "num_dec_layers": int(rl_token_cfg.num_dec_layers),
        "ff_dim": rl_token_cfg.ff_dim if rl_token_cfg.ff_dim is None else int(rl_token_cfg.ff_dim),
        "num_rl_tokens": int(getattr(rl_token, "num_rl_tokens")),
        "inference_only": bool(getattr(rl_token, "inference_only", False)),
        "normalize_encoder_output": bool(getattr(rl_token, "normalize_encoder_output", False)),
    }

    train_config = dataclasses.asdict(rl_token_cfg) if dataclasses.is_dataclass(rl_token_cfg) else dict(rl_token_cfg)
    meta = {
        "format_version": RL_TOKEN_META_FORMAT_VERSION,
        "module": "evo_rlt.core.rl_token.RLTokenModule",
        "train_config": train_config,
        "module_config": module_config,
        "postprocess": {
            "image_only": bool(rl_token_cfg.image_only),
            "token_pool_size": int(rl_token_cfg.token_pool_size),
            "num_image_tokens": None if num_image_tokens is None else int(num_image_tokens),
        },
    }
    with open(rl_token_dir / RL_TOKEN_META_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)


def load_rl_token_state(checkpoint_dir: Path, rl_token: nn.Module, strict: bool = True) -> dict:
    rl_token_dir = checkpoint_dir / RL_TOKEN_DIR
    state_path = rl_token_dir / RL_TOKEN_STATE_FILENAME
    meta_path = rl_token_dir / RL_TOKEN_META_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"Missing rl_token state: {state_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing rl_token meta: {meta_path}")

    state_dict = load_file(str(state_path))
    target_device = next(rl_token.parameters()).device
    rl_token.load_state_dict({k: v.to(target_device) for k, v in state_dict.items()}, strict=strict)
    with open(meta_path) as f:
        return json.load(f)
