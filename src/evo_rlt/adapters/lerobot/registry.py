from __future__ import annotations

from typing import Any

_REGISTERED = False


def register() -> None:
    """Register Evo-RLT policy/config factories with LeRobot at runtime.

    This keeps Evo-RLT out of LeRobot source files. Call once before using
    LeRobot factory helpers with policy.type in {rlt, rlt_token, rlt_ac}.
    """
    from evo_rlt.adapters.lerobot.pi05_image_preprocessing import install_pi05_openpi_resize

    install_pi05_openpi_resize()

    global _REGISTERED
    if _REGISTERED:
        return

    import lerobot.policies.factory as factory

    from lerobot.configs.policies import PreTrainedConfig

    from evo_rlt.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig

    PreTrainedConfig._choice_registry["rlt"] = RLTPretrainedConfig
    PreTrainedConfig._choice_registry["rlt_ac"] = ChunkACPolicyConfig
    PreTrainedConfig._choice_registry["rlt_token"] = RLTokenPolicyConfig

    original_get_policy_class = factory.get_policy_class
    original_make_policy_config = factory.make_policy_config
    original_make_pre_post_processors = factory.make_pre_post_processors

    def get_policy_class(name: str):
        if name == "rlt_token":
            from evo_rlt.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy

            return RLTokenPolicy
        if name == "rlt_ac":
            from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy

            return ChunkACPolicy
        if name == "rlt":
            from evo_rlt.adapters.lerobot.policies.modeling_rlt import RLTPretrainedPolicy

            return RLTPretrainedPolicy
        return original_get_policy_class(name)

    def make_policy_config(policy_type: str, **kwargs: Any):
        if policy_type == "rlt_token":
            return RLTokenPolicyConfig(**kwargs)
        if policy_type == "rlt_ac":
            return ChunkACPolicyConfig(**kwargs)
        if policy_type == "rlt":
            return RLTPretrainedConfig(**kwargs)
        return original_make_policy_config(policy_type, **kwargs)

    def make_pre_post_processors(policy_cfg, *args: Any, **kwargs: Any):
        if isinstance(policy_cfg, RLTokenPolicyConfig):
            from evo_rlt.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors

            return make_rlt_token_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        if isinstance(policy_cfg, ChunkACPolicyConfig):
            from evo_rlt.adapters.lerobot.policies.processor_rlt_ac import make_rlt_ac_pre_post_processors

            return make_rlt_ac_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return original_make_pre_post_processors(policy_cfg, *args, **kwargs)

    factory.get_policy_class = get_policy_class
    factory.make_policy_config = make_policy_config
    factory.make_pre_post_processors = make_pre_post_processors
    _REGISTERED = True
