from contextlib import contextmanager
from types import SimpleNamespace

from transformers import initialization as transformers_initialization

from evo_rlt.adapters.lerobot.record import backend


def test_pretrained_pi05_policy_uses_no_init_weights(monkeypatch):
    events = []
    expected_policy = object()
    dataset_meta = object()

    @contextmanager
    def fake_no_init_weights():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def fake_make_policy(policy_cfg, ds_meta):
        assert policy_cfg.type == "pi05"
        assert ds_meta is dataset_meta
        assert events == ["enter"]
        return expected_policy

    monkeypatch.setattr(transformers_initialization, "no_init_weights", fake_no_init_weights)
    monkeypatch.setattr(backend, "make_policy", fake_make_policy)

    policy_cfg = SimpleNamespace(type="pi05", pretrained_path="/tmp/model")
    actual_policy = backend._make_recording_policy(policy_cfg, dataset_meta)

    assert actual_policy is expected_policy
    assert events == ["enter", "exit"]


def test_non_pi05_policy_uses_normal_initialization(monkeypatch):
    expected_policy = object()
    dataset_meta = object()

    def fail_no_init_weights():
        raise AssertionError("non-PI0.5 policy must not use no_init_weights")

    def fake_make_policy(policy_cfg, ds_meta):
        assert policy_cfg.type == "diffusion"
        assert ds_meta is dataset_meta
        return expected_policy

    monkeypatch.setattr(transformers_initialization, "no_init_weights", fail_no_init_weights)
    monkeypatch.setattr(backend, "make_policy", fake_make_policy)

    policy_cfg = SimpleNamespace(type="diffusion", pretrained_path="/tmp/model")
    actual_policy = backend._make_recording_policy(policy_cfg, dataset_meta)

    assert actual_policy is expected_policy


def test_missing_policy_stays_none(monkeypatch):
    def fail_make_policy(*args, **kwargs):
        raise AssertionError("make_policy must not be called without a policy config")

    monkeypatch.setattr(backend, "make_policy", fail_make_policy)

    assert backend._make_recording_policy(None, object()) is None
