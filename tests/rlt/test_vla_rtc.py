from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import torch

from evo_rlt.adapters.lerobot.record import vla_rtc


class _InlineThread:
    def __init__(self, *, target, args, daemon):
        del daemon
        self._target = target
        self._args = args
        self._alive = False

    def start(self) -> None:
        self._alive = True
        try:
            self._target(*self._args)
        finally:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self) -> None:
        return None


class _FakePolicy:
    def __init__(self, *, chunk_size: int, execution_horizon: int) -> None:
        self.config = SimpleNamespace(
            chunk_size=chunk_size,
            rtc_config=SimpleNamespace(enabled=True, execution_horizon=execution_horizon),
        )
        self.chunk_calls = 0
        self.chunk_kwargs = []
        self.consume_during_prediction = 0
        self.runtime = None

    def predict_action_chunk(self, batch, **kwargs):
        del batch
        self.chunk_calls += 1
        self.chunk_kwargs.append(kwargs)
        if self.runtime is not None:
            for _ in range(self.consume_during_prediction):
                self.runtime.action_queue.get()
        return torch.arange(self.config.chunk_size, dtype=torch.float32).reshape(1, -1, 1)


def test_guidance_delay_uses_bootstrap_once_then_recent_real_delays() -> None:
    tracker = vla_rtc._GuidanceDelayTracker(window_size=3)  # noqa: SLF001

    tracker.observe_bootstrap(latency=2.0, fps=30.0)
    assert tracker.current() == 60

    tracker.observe_real_delay(14)
    assert tracker.current() == 14
    tracker.observe_real_delay(8)
    tracker.observe_real_delay(8)
    assert tracker.current() == 14

    tracker.observe_real_delay(8)
    assert tracker.current() == 8


def test_runtime_replaces_bootstrap_delay_with_recent_real_queue_delays() -> None:
    policy = _FakePolicy(chunk_size=6, execution_horizon=3)
    runtime = vla_rtc.VLAOnlyRTCRuntime(
        policy,
        fps=30.0,
        device=torch.device("cpu"),
        use_amp=False,
        inference_lock=Lock(),
    )
    policy.runtime = runtime
    runtime.guidance_delay_tracker = vla_rtc._GuidanceDelayTracker(window_size=2)  # noqa: SLF001
    runtime.guidance_delay_tracker.observe_bootstrap(latency=2.0, fps=30.0)
    initial_chunk = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
    runtime.action_queue.merge(initial_chunk, initial_chunk, 0, 0)

    for real_delay in (3, 1, 1):
        request = runtime._prepare_request({"observation.state": torch.zeros(1)})  # noqa: SLF001
        policy.consume_during_prediction = real_delay
        runtime._predict_and_merge(*request)  # noqa: SLF001

    final_request = runtime._prepare_request({"observation.state": torch.zeros(1)})  # noqa: SLF001

    assert [kwargs["inference_delay"] for kwargs in policy.chunk_kwargs] == [60, 3, 3]
    assert final_request[2] == 1


def test_runtime_starts_background_refill_every_execution_horizon_actions(monkeypatch) -> None:
    monkeypatch.setattr(vla_rtc, "Thread", _InlineThread)
    policy = _FakePolicy(chunk_size=6, execution_horizon=3)
    runtime = vla_rtc.VLAOnlyRTCRuntime(
        policy,
        fps=30.0,
        device=torch.device("cpu"),
        use_amp=False,
        inference_lock=Lock(),
    )

    calls_after_action = []
    for _ in range(6):
        runtime.select_action({"observation.state": torch.zeros(1)})
        calls_after_action.append(policy.chunk_calls)

    runtime.close()
    assert calls_after_action == [1, 1, 2, 2, 2, 3]
