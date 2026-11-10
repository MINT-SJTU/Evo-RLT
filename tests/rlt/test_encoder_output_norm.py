from __future__ import annotations

import io

import pytest
import torch

from evo_rlt.adapters.lerobot.train_utils import load_rl_token_state, save_rl_token_state
from evo_rlt.adapters.lerobot.training_config import RLTokenJointConfig
from evo_rlt.core.algorithm import RLTAlgorithm
from evo_rlt.core.config import RLTConfig, RLTokenConfig
from evo_rlt.core.policy import RLTPolicy
from evo_rlt.core.rl_token import RLTokenModule
from evo_rlt.core.utils import filter_encoder_only
from evo_rlt.core.vla_adapter import DummyVLAAdapter


def make_module(normalize=True, inference_only=False):
    return RLTokenModule(
        token_dim=16, nhead=4, num_enc_layers=1, num_dec_layers=1,
        ff_dim=32, num_rl_tokens=2, inference_only=inference_only,
        normalize_encoder_output=normalize,
    )


def make_algorithm():
    config = RLTConfig()
    config.rl_token = RLTokenConfig(
        token_dim=16, nhead=4, enc_layers=1, dec_layers=1,
        ff_dim=32, num_rl_tokens=2, normalize_encoder_output=True,
    )
    vla = DummyVLAAdapter(token_dim=16)
    return RLTAlgorithm(RLTPolicy(config, vla), config)


def test_normalized_bottleneck_handles_large_encoder_outputs():
    torch.manual_seed(0)
    module = make_module().eval()
    # Mimic a large residual-stream scale and offset at the encoder output.
    handle = module.encoder.register_forward_hook(lambda _m, _i, out: out * 1000 + 500)
    tokens = torch.randn(2, 8, 16)
    try:
        latent = module.encode_multi(tokens)
        torch.testing.assert_close(latent.mean(-1), torch.zeros(2, 2), atol=1e-5, rtol=0)
        torch.testing.assert_close(latent.var(-1, unbiased=False), torch.ones(2, 2), atol=1e-5, rtol=0)
        torch.testing.assert_close(module.encode(tokens), latent.mean(1))
        loss = module.reconstruction_loss(tokens)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert module.encoder_out_norm.weight.grad.abs().sum() > 0
    finally:
        handle.remove()


def test_decoder_receives_the_same_normalized_latent_as_encoding():
    module = make_module().eval()
    tokens = torch.randn(2, 8, 16)
    inputs = []
    handle = module.decoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: inputs.append(kwargs['memory'].detach()), with_kwargs=True,
    )
    try:
        module.reconstruction_loss(tokens)
    finally:
        handle.remove()
    torch.testing.assert_close(inputs[0], module.encode_multi(tokens))


def test_default_preserves_legacy_keys_and_encoder_output():
    module = RLTokenModule(token_dim=16, nhead=4, num_enc_layers=1, num_dec_layers=1).eval()
    assert not module.normalize_encoder_output
    assert not any(key.startswith('encoder_out_norm.') for key in module.state_dict())
    tokens = torch.randn(2, 8, 16)
    raw = module.encoder(torch.cat([tokens, module.rl_token_embed.expand(2, -1, -1)], dim=1))
    torch.testing.assert_close(module.encode_multi(tokens), raw[:, -1:, :])
    rebuilt = RLTokenModule(token_dim=16, nhead=4, num_enc_layers=1, num_dec_layers=1).eval()
    rebuilt.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(rebuilt.encode(tokens), module.encode(tokens))


@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('strict', [False, True])
def test_checkpoint_rejects_mismatched_normalization(normalize, strict):
    source = make_module(normalize=normalize)
    destination = make_module(normalize=not normalize)
    with pytest.raises(RuntimeError, match='normalize_encoder_output'):
        destination.load_state_dict(source.state_dict(), strict=strict)


def test_partial_normalization_checkpoint_is_rejected_with_non_strict_loading():
    module = make_module()
    state = module.state_dict()
    del state['encoder_out_norm.bias']
    with pytest.raises(RuntimeError, match='both weight and bias'):
        module.load_state_dict(state, strict=False)


def test_parent_module_loading_checks_normalization_with_prefix():
    source = torch.nn.ModuleDict({'rl_token': make_module(normalize=True)})
    destination = torch.nn.ModuleDict({'rl_token': make_module(normalize=False)})
    with pytest.raises(RuntimeError, match='rl_token.normalize_encoder_output'):
        destination.load_state_dict(source.state_dict(), strict=False)


@pytest.mark.parametrize('normalize', [False, True])
def test_checkpoint_round_trip_into_inference_encoder(normalize):
    source = make_module(normalize=normalize).eval()
    if normalize:
        with torch.no_grad():
            source.encoder_out_norm.weight.fill_(1.7)
            source.encoder_out_norm.bias.fill_(.3)
    file = io.BytesIO()
    torch.save(source.state_dict(), file)
    file.seek(0)
    state, _ = filter_encoder_only(torch.load(file, weights_only=True))
    destination = make_module(normalize=normalize, inference_only=True).eval()
    destination.load_state_dict(state, strict=True)
    tokens = torch.randn(2, 8, 16)
    torch.testing.assert_close(destination.encode(tokens), source.encode(tokens))


def test_policy_sync_and_freeze_include_normalization():
    algorithm = make_algorithm()
    with torch.no_grad():
        algorithm.policy.rl_token.encoder_out_norm.weight.fill_(1.7)
        algorithm.policy.rl_token.encoder_out_norm.bias.fill_(.3)
    full = algorithm.build_rl_token_full('cpu').eval()
    assert full.normalize_encoder_output
    algorithm.policy.eval()
    tokens = torch.randn(2, 8, 16)
    torch.testing.assert_close(full.encode(tokens), algorithm.policy.rl_token.encode(tokens))
    with torch.no_grad():
        full.encoder_out_norm.weight.fill_(2.3)
        full.encoder_out_norm.bias.fill_(-.7)
    algorithm.sync_encoder_from_full(full)
    torch.testing.assert_close(full.encode(tokens), algorithm.policy.rl_token.encode(tokens))
    algorithm.policy.freeze_rl_token_encoder()
    assert all(not p.requires_grad for p in algorithm.policy.rl_token.parameters())


@pytest.mark.parametrize('normalize', [False, True])
def test_joint_checkpoint_metadata_preserves_normalization(tmp_path, normalize):
    source = make_module(normalize=normalize).eval()
    config = RLTokenJointConfig(nhead=4, num_enc_layers=1, num_dec_layers=1, ff_dim=32)
    save_rl_token_state(tmp_path, source, config)
    destination = make_module(normalize=normalize).eval()
    metadata = load_rl_token_state(tmp_path, destination)
    assert metadata['module_config']['normalize_encoder_output'] is normalize
    rebuilt = RLTokenModule(**metadata['module_config']).eval()
    load_rl_token_state(tmp_path, rebuilt)
    tokens = torch.randn(2, 8, 16)
    torch.testing.assert_close(rebuilt.encode(tokens), source.encode(tokens))


def test_core_yaml_can_enable_normalization(tmp_path):
    path = tmp_path / 'normalized.yaml'
    path.write_text('rl_token:\n  normalize_encoder_output: true\n')
    assert RLTConfig.from_yaml(path).rl_token.normalize_encoder_output
