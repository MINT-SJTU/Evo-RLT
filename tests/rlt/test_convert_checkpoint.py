from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from evo_rlt.adapters.lerobot.convert_checkpoint import convert
from evo_rlt.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt import RLTPretrainedPolicy
from evo_rlt.core.rl_token import RLTokenModule


@pytest.mark.parametrize("normalize, embedded_encoder", [(True, False), (False, True), (True, True)])
def test_convert_checkpoint_round_trip(tmp_path, normalize, embedded_encoder):
    source = RLTPretrainedPolicy(RLTPretrainedConfig(
        rl_token_dim=16, rl_token_nhead=4, rl_token_enc_layers=1,
        rl_token_ff_dim=32, rl_token_num_rl_tokens=2,
        rl_token_normalize_encoder_output=normalize,
        actor_hidden_dim=16, actor_num_layers=2,
        action_dim=3, proprio_dim=3, chunk_length=2, device="cpu",
    )).eval()
    if normalize:
        with torch.no_grad():
            source.rl_token.encoder_out_norm.bias.fill_(0.5)
            source.rl_token.encoder_out_norm.weight.fill_(1.5)

    demo_path = tmp_path / "demo.pt"
    actor_path = tmp_path / "actor.pt"
    actor_checkpoint = {
        "actor_state_dict": source.actor.state_dict(),
        "metadata": {
            "actor": {"hidden_dim": 16, "num_layers": 2},
            "action_dim": 3,
            "proprio_dim": 3,
            "chunk_length": 2,
        },
    }
    demo_encoder = RLTokenModule(
        token_dim=16, nhead=4, num_enc_layers=1, num_dec_layers=1,
        ff_dim=32, num_rl_tokens=2,
        normalize_encoder_output=not normalize if embedded_encoder else normalize,
    )
    if embedded_encoder:
        actor_checkpoint["rl_token_state_dict"] = source.rl_token.state_dict()
    else:
        demo_encoder.load_state_dict(source.rl_token.state_dict(), strict=False)
    torch.save({"rl_token_state_dict": demo_encoder.state_dict()}, demo_path)
    torch.save(actor_checkpoint, actor_path)

    out = convert(
        str(demo_path), str(actor_path), str(tmp_path / "converted"),
        config_overrides={"device": "cpu", "rl_token_nhead": 4},
    )
    demo_path.unlink()
    actor_path.unlink()
    restored = RLTPretrainedPolicy.from_pretrained(out)

    assert restored.config.rl_token_normalize_encoder_output is normalize
    assert not restored.config.rl_token_ckpt_path
    assert not restored.config.ac_ckpt_path
    torch.testing.assert_close(restored.state_dict(), source.state_dict())
    tokens = torch.randn(2, 8, 16)
    with torch.no_grad():
        torch.testing.assert_close(restored.rl_token.encode(tokens), source.rl_token.encode(tokens))


def test_conversion_requires_an_encoder(tmp_path):
    actor_path = tmp_path / "actor.pt"
    torch.save({"actor_state_dict": {}}, actor_path)
    out = tmp_path / "converted"
    with pytest.raises(ValueError, match="--rl-token-ckpt"):
        convert(None, str(actor_path), str(out), {"device": "cpu"})
    assert not out.exists()


def test_loading_rejects_checkpoint_without_deployment_weight_keys(tmp_path):
    config = RLTPretrainedConfig(
        rl_token_dim=16, rl_token_nhead=4, rl_token_enc_layers=1,
        rl_token_ff_dim=32, actor_hidden_dim=16, device="cpu",
    )
    policy = RLTPretrainedPolicy(config)
    legacy_weights = {
        key.removeprefix("modifier."): tensor for key, tensor in policy.state_dict().items()
    }
    config.save_pretrained(tmp_path)
    save_file(legacy_weights, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="modifier.rl_token"):
        RLTPretrainedPolicy.from_pretrained(tmp_path)
