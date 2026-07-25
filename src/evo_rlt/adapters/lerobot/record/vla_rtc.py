"""Asynchronous Real-Time Chunking runtime for VLA-only policies."""

from __future__ import annotations

import copy
import logging
import math
import time
from contextlib import nullcontext
from threading import Lock, Thread
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker

log = logging.getLogger(__name__)

_RUNTIMES_ATTRIBUTE = "_evo_rlt_vla_rtc_runtimes"
_INFERENCE_LOCK_ATTRIBUTE = "_evo_rlt_vla_rtc_inference_lock"


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            cloned[key] = value.detach().clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _get_rtc_processor(policy: PreTrainedPolicy) -> Any | None:
    processor = getattr(policy, "rtc_processor", None)
    if processor is not None:
        return processor
    return getattr(getattr(policy, "model", None), "rtc_processor", None)


class VLAOnlyRTCRuntime:
    """Overlap PI0.5 inference with action execution and merge chunks using RTC."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        *,
        fps: float,
        device: torch.device,
        use_amp: bool,
        inference_lock: Lock,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        rtc_config = getattr(policy.config, "rtc_config", None)
        if rtc_config is None or not rtc_config.enabled:
            raise ValueError("VLAOnlyRTCRuntime requires an enabled RTC config")

        chunk_size = int(policy.config.chunk_size)
        execution_horizon = int(rtc_config.execution_horizon)
        if not 0 < execution_horizon <= chunk_size:
            raise ValueError(
                f"RTC execution_horizon must be in [1, {chunk_size}], got {execution_horizon}"
            )

        self.policy = policy
        self.fps = float(fps)
        self.device = device
        self.use_amp = use_amp
        self.rtc_config = rtc_config
        self.refill_threshold = max(1, chunk_size - execution_horizon)
        self.action_queue = ActionQueue(rtc_config)
        self.latency_tracker = LatencyTracker()
        self.inference_lock = inference_lock
        self.worker: Thread | None = None
        self.worker_error: Exception | None = None
        self.state_lock = Lock()

        log.info(
            "[VLA RTC] runtime initialized | chunk_size=%d execution_horizon=%d "
            "refill_threshold=%d fps=%.1f",
            chunk_size,
            execution_horizon,
            self.refill_threshold,
            self.fps,
        )

    def _autocast_context(self):
        if self.device.type == "cuda" and self.use_amp:
            return torch.autocast(device_type=self.device.type)
        return nullcontext()

    def _raise_worker_error(self) -> None:
        if self.worker_error is None:
            return
        error = self.worker_error
        self.worker_error = None
        raise error

    def _join_finished_worker(self) -> None:
        if self.worker is None or self.worker.is_alive():
            return
        self.worker.join()
        self.worker = None
        self._raise_worker_error()

    def _wait_for_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.join()
        self.worker = None
        self._raise_worker_error()

    def _prepare_request(self, batch: dict[str, Any]) -> tuple[dict[str, Any], Tensor | None, int, int, float]:
        with self.state_lock:
            previous_actions = self.action_queue.get_left_over()
            action_index = self.action_queue.get_action_index()
        max_latency = self.latency_tracker.max() or 0.0
        inference_delay = math.ceil(max_latency * self.fps)
        return _clone_batch(batch), previous_actions, inference_delay, action_index, time.perf_counter()

    def _debug_metrics(self, previous_actions: Tensor | None) -> tuple[str, float | None, float | None]:
        if previous_actions is None:
            return "bootstrap", None, None

        processor = _get_rtc_processor(self.policy)
        if processor is None or not processor.is_debug_enabled():
            return "applied", None, None

        steps = processor.get_all_debug_steps()
        corrections = [step.correction for step in steps if step.correction is not None]
        guidance_weights = [step.guidance_weight for step in steps if step.guidance_weight is not None]
        if not corrections:
            raise RuntimeError("RTC debug is enabled but no guidance correction was recorded")

        correction_norm = max(float(correction.norm().item()) for correction in corrections)
        guidance_weight = max(
            float(weight.item()) if isinstance(weight, Tensor) else float(weight)
            for weight in guidance_weights
        )
        return "applied", correction_norm, guidance_weight

    def _predict_and_merge(
        self,
        batch: dict[str, Any],
        previous_actions: Tensor | None,
        inference_delay: int,
        action_index_before_inference: int,
        request_start_time: float,
    ) -> None:
        with self.inference_lock, torch.no_grad(), self._autocast_context():
            processor = _get_rtc_processor(self.policy)
            if processor is not None and processor.is_debug_enabled():
                processor.reset_tracker()
            chunk = self.policy.predict_action_chunk(
                batch,
                inference_delay=inference_delay,
                prev_chunk_left_over=previous_actions,
                execution_horizon=self.rtc_config.execution_horizon,
            )
            guidance, correction_norm, guidance_weight = self._debug_metrics(previous_actions)

        if chunk.shape[0] != 1:
            raise ValueError(f"RTC deployment expects batch size 1, got {chunk.shape[0]}")
        actions = chunk.squeeze(0).detach()
        latency = time.perf_counter() - request_start_time
        self.latency_tracker.add(latency)

        with self.state_lock:
            real_delay = max(0, self.action_queue.get_action_index() - action_index_before_inference)
            queue_before_merge = self.action_queue.qsize()
            self.action_queue.merge(actions, actions, real_delay, action_index_before_inference)

        correction_text = "n/a" if correction_norm is None else f"{correction_norm:.6f}"
        weight_text = "n/a" if guidance_weight is None else f"{guidance_weight:.3f}"
        log.info(
            "[VLA RTC] chunk merged | latency=%.1fms estimated_delay=%d real_delay=%d "
            "inference_delay=%d prev_leftover=%d queue_before_merge=%d guidance=%s "
            "correction_norm_max=%s guidance_weight_max=%s",
            latency * 1000.0,
            math.ceil(latency * self.fps),
            real_delay,
            inference_delay,
            0 if previous_actions is None else previous_actions.shape[0],
            queue_before_merge,
            guidance,
            correction_text,
            weight_text,
        )

    def _run_worker(self, request: tuple[dict[str, Any], Tensor | None, int, int, float]) -> None:
        try:
            self._predict_and_merge(*request)
        except Exception as error:
            with self.state_lock:
                self.worker_error = error

    def _maybe_start_worker(self, batch: dict[str, Any]) -> None:
        self._join_finished_worker()
        if self.worker is not None or self.action_queue.qsize() > self.refill_threshold:
            return
        request = self._prepare_request(batch)
        self.worker = Thread(target=self._run_worker, args=(request,), daemon=True)
        self.worker.start()

    def select_action(self, batch: dict[str, Any]) -> Tensor:
        self._join_finished_worker()
        if self.action_queue.empty():
            self._wait_for_worker()
        if self.action_queue.empty():
            self._predict_and_merge(*self._prepare_request(batch))

        with self.state_lock:
            action = self.action_queue.get()
        if action is None:
            raise RuntimeError("RTC action queue is empty after refill")
        self._maybe_start_worker(batch)
        return action.unsqueeze(0)

    def close(self) -> None:
        self._wait_for_worker()


def get_vla_only_rtc_runtime(
    policy: PreTrainedPolicy,
    *,
    stream_id: str | None,
    fps: float,
    device: torch.device,
    use_amp: bool,
) -> VLAOnlyRTCRuntime:
    runtimes = getattr(policy, _RUNTIMES_ATTRIBUTE, None)
    if runtimes is None:
        runtimes = {}
        setattr(policy, _RUNTIMES_ATTRIBUTE, runtimes)

    runtime = runtimes.get(stream_id)
    if runtime is not None:
        if runtime.fps != float(fps):
            raise ValueError(f"RTC runtime fps changed from {runtime.fps} to {fps}")
        return runtime

    inference_lock = getattr(policy, _INFERENCE_LOCK_ATTRIBUTE, None)
    if inference_lock is None:
        inference_lock = Lock()
        setattr(policy, _INFERENCE_LOCK_ATTRIBUTE, inference_lock)
    runtime = VLAOnlyRTCRuntime(
        policy,
        fps=fps,
        device=device,
        use_amp=use_amp,
        inference_lock=inference_lock,
    )
    runtimes[stream_id] = runtime
    return runtime


def reset_vla_only_rtc_runtimes(policy: PreTrainedPolicy) -> None:
    runtimes = getattr(policy, _RUNTIMES_ATTRIBUTE, None)
    if runtimes is None:
        return
    for runtime in runtimes.values():
        runtime.close()
    runtimes.clear()
