from __future__ import annotations

from collections import deque
from threading import Lock
from types import SimpleNamespace

import torch

from evo_rlt.adapters.lerobot.policies.action_modifier import RLTStepMetadata
from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker


def _make_policy(chunk: torch.Tensor | None = None) -> ChunkACPolicy:
    policy = object.__new__(ChunkACPolicy)
    rtc_config = RTCConfig(enabled=True, execution_horizon=2)
    policy._rtc_config = rtc_config
    policy._vla_rtc_config = rtc_config
    policy._active_pi05_rtc_config = rtc_config
    policy._rtc_action_queue = ActionQueue(rtc_config)
    policy._rtc_latency_tracker = LatencyTracker()
    policy._rtc_fps = 10.0
    policy._rtc_action_queue_size_to_get_new_actions = 1
    policy._rtc_worker = None
    policy._rtc_worker_error = None
    policy._rtc_generation = 0
    policy._rtc_lock = Lock()
    policy._rtc_inference_lock = Lock()
    policy._rtc_step_metadata = deque()
    policy._rtc_selected_step_metadata = deque()
    policy.predict_calls = []
    if chunk is None:
        chunk = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    def predict_action_chunk(batch, **kwargs):
        policy.predict_calls.append((batch, kwargs))
        metadata = [RLTStepMetadata(phase=1.0, source_type=1.0) for _ in range(chunk.shape[1])]
        policy._ensure_modifier()._step_metadata.extend(metadata)
        return chunk.clone()

    object.__setattr__(policy, "predict_action_chunk", predict_action_chunk)
    object.__setattr__(policy, "_ensure_modifier", lambda: policy.modifier)
    policy.modifier = type("_Modifier", (), {"_step_metadata": deque()})()
    return policy


def test_configure_rtc_accepts_vla_rtc_config_and_switches_by_phase() -> None:
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(chunk_length=10, action_dim=3, proprio_dim=3)
    policy._rtc_lock = Lock()
    policy._rtc_inference_lock = Lock()
    policy._rtc_step_metadata = deque()
    policy._rtc_selected_step_metadata = deque()
    policy.vla_seen_configs = []

    class FakePi05:
        def __init__(self) -> None:
            self.config = SimpleNamespace(rtc_config=None)
            self.init_calls: list[RTCConfig] = []

        def init_rtc_processor(self) -> None:
            self.init_calls.append(self.config.rtc_config)

        def predict_action_chunk(self, batch, **kwargs):
            policy.vla_seen_configs.append(self.config.rtc_config)
            return torch.zeros(1, 4, 3)

    fake_pi05 = FakePi05()
    policy._rl_token_policy = SimpleNamespace(_pi05=fake_pi05)
    policy._prefix_capture = SimpleNamespace(consume=lambda: torch.zeros(1, 1, 1))
    policy.modifier = SimpleNamespace(
        is_rl_phase=False,
        compute_chunk=lambda vla_chunk, proprio, prefix_tokens: vla_chunk,
    )
    object.__setattr__(policy, "eval", lambda: policy)
    object.__setattr__(policy, "_ensure_modifier", lambda: policy.modifier)

    rlt_rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    vla_rtc_config = RTCConfig(enabled=True, execution_horizon=25)
    policy.configure_rtc(rlt_rtc_config, fps=30, vla_rtc_config=vla_rtc_config)

    policy.predict_action_chunk({"observation.state": torch.ones(1, 3)})
    policy.modifier.is_rl_phase = True
    policy.predict_action_chunk({"observation.state": torch.ones(1, 3)})

    assert fake_pi05.init_calls == [rlt_rtc_config, vla_rtc_config, rlt_rtc_config]
    assert policy.vla_seen_configs == [vla_rtc_config, rlt_rtc_config]


def test_prepare_rtc_request_uses_leftover_and_latency() -> None:
    policy = _make_policy()
    old_actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    obs = torch.ones(1, 3)
    policy._rtc_action_queue.merge(old_actions, old_actions, real_delay=0)
    policy._rtc_action_queue.get()
    policy._rtc_latency_tracker.add(0.21)

    batch, prev_actions, inference_delay, action_index, _, _ = policy._prepare_rtc_request(
        {"observation.state": obs, "task": ["insert"]}
    )

    assert action_index == 1
    assert inference_delay == 3
    assert torch.equal(prev_actions, old_actions[1:])
    assert batch["task"] == ["insert"]
    assert torch.equal(batch["observation.state"], obs)
    assert batch["observation.state"] is not obs


def test_predict_and_merge_rtc_chunk_uses_actual_consumed_delay() -> None:
    policy = _make_policy()
    old_actions = torch.full((4, 3), -1.0)
    policy._rtc_action_queue.merge(old_actions, old_actions, real_delay=0)
    action_index_before_inference = policy._rtc_action_queue.get_action_index()
    policy._rtc_action_queue.get()
    policy._rtc_action_queue.get()

    policy._predict_and_merge_rtc_chunk(
        {"observation.state": torch.ones(1, 3)},
        prev_actions=old_actions[2:],
        inference_delay=1,
        action_index_before_inference=action_index_before_inference,
        request_start_time=0.0,
        generation=0,
    )

    assert policy._rtc_action_queue.qsize() == 2
    assert torch.equal(policy._rtc_action_queue.get(), torch.tensor([6.0, 7.0, 8.0]))
    assert len(policy._rtc_step_metadata) == 2
    _, kwargs = policy.predict_calls[0]
    assert kwargs["inference_delay"] == 1
    assert torch.equal(kwargs["prev_chunk_left_over"], old_actions[2:])


def test_select_action_rtc_returns_matching_metadata() -> None:
    policy = _make_policy()
    policy._rtc_latency_tracker.add(1.0)

    action = policy._select_action_rtc({"observation.state": torch.ones(1, 3)})
    metadata = policy.pop_step_metadata()

    assert action.shape == (1, 3)
    assert torch.equal(action, torch.tensor([[0.0, 1.0, 2.0]]))
    assert metadata == RLTStepMetadata(phase=1.0, source_type=1.0)
    assert policy._rtc_action_queue.qsize() == 3


def test_reset_rtc_runtime_invalidates_inflight_request_without_joining() -> None:
    policy = _make_policy()
    request = policy._prepare_rtc_request({"observation.state": torch.ones(1, 3)})
    policy._reset_rtc_runtime()

    policy._predict_and_merge_rtc_chunk(*request)

    assert policy._rtc_action_queue.qsize() == 0
    assert len(policy._rtc_step_metadata) == 0
    assert policy.predict_calls == []


def test_replay_encoding_serializes_vla_access_without_touching_rtc_queues():
    policy = _make_policy()
    policy.config = SimpleNamespace(chunk_length=3, action_dim=2, proprio_dim=2)
    policy.modifier.encode_replay_state = lambda proprio, prefix: torch.cat([prefix[:, 0], proprio], dim=-1)
    original_metadata = RLTStepMetadata(phase=1., source_type=1.)
    policy.modifier._step_metadata.append(original_metadata)
    policy._rtc_action_queue.merge(torch.ones(4, 2), torch.ones(4, 2), real_delay=0)

    class Pi05:
        def predict_action_chunk(self, batch):
            assert policy._rtc_inference_lock.locked()
            return torch.full((1, 4, 2), 7.)

    policy._rl_token_policy = SimpleNamespace(_pi05=Pi05())
    policy._prefix_capture = SimpleNamespace(consume=lambda: torch.full((1, 1, 3), 5.))
    object.__setattr__(policy, 'eval', lambda: policy)
    state, ref = policy.encode_replay_observation({'observation.state': torch.tensor([[1., 2.]])})
    assert torch.equal(state, torch.tensor([[5., 5., 5., 1., 2.]]))
    assert torch.equal(ref, torch.full((1, 3, 2), 7.))
    assert list(policy.modifier._step_metadata) == [original_metadata]
    assert policy._rtc_action_queue.qsize() == 4
