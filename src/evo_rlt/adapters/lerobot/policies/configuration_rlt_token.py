from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

_DEFAULT_CAMERA_KEYS: list[str] = ["left_wrist", "right_wrist", "right_front"]


@dataclass
class RLTokenPolicyConfig(PreTrainedConfig):
    """Config for the RL Token training policy.

    Trains an RLTokenModule (encoder + decoder) that reconstructs pi0.5's prefix
    hidden states. The frozen pi0.5 backbone is held in __dict__ at runtime so it
    does not end up in the saved safetensors. Save layout matches SFT pi0.5:
    config.json + model.safetensors (only rl_token weights) +
    policy_preprocessor.json + policy_postprocessor.json + train_config.json.
    """

    # --- VLA backbone (frozen, not serialized) ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_revision: str | None = None
    vla_dtype: str = "bfloat16"

    # --- RL Token encoder + decoder ---
    rl_token_dim: int = 2048
    rl_token_nhead: int = 8
    rl_token_enc_layers: int = 3
    rl_token_dec_layers: int = 3
    rl_token_ff_dim: int = 4096
    rl_token_num_rl_tokens: int = 1
    rl_token_init_scale: float = 0.02
    rl_token_normalize_encoder_output: bool = False

    # --- Prefix token postprocessing (before RL token encoder) ---
    token_pool_size: int = 0
    image_only: bool = False
    active_camera_indices: list[int] | None = None
    num_per_camera: int = 0

    # --- Loss weights ---
    recon_weight: float = 1.0
    vla_ft_weight: float = 0.0
    norm_gamma: float = 0.0
    norm_stats_path: str | None = None

    # --- Per-group learning rates (joint training; used when vla_ft_weight>0) ---
    # These override --optimizer.lr / --scheduler.peak_lr for their respective
    # param group. The cosine scheduler still scales both groups by the same
    # LambdaLR factor, so the RL-token : VLA ratio is preserved end-to-end.
    rl_token_lr: float = 2e-4
    vla_lr: float = 2e-5

    # --- Dimensions ---
    chunk_size: int = 50
    action_dim: int = 12
    proprio_dim: int = 12

    # --- Observation mapping ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))

    # --- Normalization MATCHES PI05 — this is the deploy-bug fix ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # --- pi05 proxy fields (used by make_rlt_token_pre_post_processors) ---
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer_path: str = "google/paligemma-3b-pt-224"

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        PreTrainedConfig._choice_registry["rlt_token"] = cls

    def __post_init__(self) -> None:
        self.ensure_registered()
        super().__post_init__()
        if self.vla_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"vla_dtype must be 'bfloat16' or 'float32', got {self.vla_dtype!r}")
        if self.recon_weight == 0 and self.vla_ft_weight == 0:
            raise ValueError("recon_weight and vla_ft_weight cannot both be zero")

    @property
    def type(self) -> str:
        return "rlt_token"

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.proprio_dim,)
            )
        for cam_key in self.camera_keys:
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, *self.image_resolution)
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.action_dim,)
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=2e-4, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=2e-4,
            decay_lr=5e-6,
            num_warmup_steps=200,
            num_decay_steps=5000,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
