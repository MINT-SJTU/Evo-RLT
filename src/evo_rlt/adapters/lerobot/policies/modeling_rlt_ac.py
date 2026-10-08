from __future__ import annotations

import copy
import logging
import math
import time
from collections import deque
from threading import Lock, Thread
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from evo_rlt.adapters.lerobot.policies.action_modifier import PrefixOutputCapture, RLTActionModifier
from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.losses import actor_loss, critic_loss
from evo_rlt.core.phase_controller import PhaseController
from evo_rlt.core.utils import soft_update

log = logging.getLogger(__name__)


class ChunkACPolicy(PreTrainedPolicy):
    """Chunk-level TD3+BC offline RL policy with VLA reference.

    Holds two frozen backbones in __dict__ (so they don't land in safetensors):
      * `_rl_token_policy`: an RLTokenPolicy loaded from
        config.rl_token_pretrained_path. It internally holds the frozen pi0.5
        backbone, so we only need one stash for the whole VLA chain.

    Trainable modules (registered as nn submodules → in safetensors + optimizer):
      * `actor`: ChunkActor — refines or replaces the VLA reference chunk.
      * `critic`: TwinCritic — twin Q networks.
      * `target_critic`: TwinCritic — Polyak-averaged copy of critic.

    forward(batch) returns one scalar loss. Each forward == one critic update.
    Actor is updated every `actor_update_interval` calls. Target critic is
    soft-updated with `tau` after every critic step. UTD=k is achieved by
    setting lerobot-train `--steps` to outer*k.
    """

    config_class = ChunkACPolicyConfig
    name = "rlt_ac"

    def __init__(
        self,
        config: ChunkACPolicyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: ChunkACPolicyConfig = config

        rl_token_policy = self._load_rl_token_policy()
        object.__setattr__(self, "_rl_token_policy", rl_token_policy)
        self._validate_rl_token_arch(rl_token_policy)

        state_dim = config.rl_token_dim + config.proprio_dim
        chunk_dim = config.chunk_length * config.action_dim

        self.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
            residual=config.actor_residual,
        )
        self.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            activation=config.critic_activation,
            layer_norm=config.critic_layer_norm,
            residual=config.critic_residual,
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.target_critic.eval()

        # Persistent step counter — survives ckpt save/load.
        self.register_buffer("_critic_step", torch.zeros((), dtype=torch.long), persistent=True)

        # Deploy-only: lazy build at .reset() time.
        self.modifier: RLTActionModifier | None = None
        # Deploy toggle (set by lerobot_rlt_record). False => the actor sees a
        # zeroed VLA reference chunk in RL phase (mirrors training ref-dropout).
        self.vla_ref: bool = True
        self._rtc_config: RTCConfig | None = None
        self._vla_rtc_config: RTCConfig | None = None
        self._active_pi05_rtc_config: RTCConfig | None = None
        self._rtc_action_queue: ActionQueue | None = None
        self._rtc_latency_tracker: LatencyTracker | None = None
        self._rtc_fps: float = 30.0
        self._rtc_action_queue_size_to_get_new_actions: int = 0
        self._rtc_worker: Thread | None = None
        self._rtc_worker_error: Exception | None = None
        self._rtc_generation: int = 0
        self._rtc_lock = Lock()
        self._rtc_inference_lock = Lock()
        self._rtc_step_metadata: deque[Any] = deque()
        self._rtc_selected_step_metadata: deque[Any] = deque()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_rl_token_policy(self) -> RLTokenPolicy:
        if not self.config.rl_token_pretrained_path:
            raise ValueError(
                "ChunkACPolicy requires config.rl_token_pretrained_path "
                "pointing at a saved RLTokenPolicy checkpoint dir."
            )
        policy = RLTokenPolicy.from_pretrained(self.config.rl_token_pretrained_path)
        for p in policy.parameters():
            p.requires_grad = False
        policy.eval()
        return policy

    def _validate_rl_token_arch(self, rl_token_policy: RLTokenPolicy) -> None:
        rtp_cfg = rl_token_policy.config
        if rtp_cfg.rl_token_dim != self.config.rl_token_dim:
            raise ValueError(
                f"rl_token_dim mismatch: ChunkACPolicyConfig={self.config.rl_token_dim} vs "
                f"RLTokenPolicy ckpt={rtp_cfg.rl_token_dim}"
            )
        if rtp_cfg.rl_token_num_rl_tokens != self.config.rl_token_num_rl_tokens:
            raise ValueError(
                f"num_rl_tokens mismatch: ChunkACPolicyConfig={self.config.rl_token_num_rl_tokens} vs "
                f"RLTokenPolicy ckpt={rtp_cfg.rl_token_num_rl_tokens}"
            )

    # ------------------------------------------------------------------
    # Training: forward
    # ------------------------------------------------------------------

    def _coerce_batch(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert ChunkTransition-style batch into the dict expected by losses.

        Required keys: state_vec, exec_chunk, ref_chunk, reward_seq,
        next_state_vec, next_ref_chunk, done, actual_steps.
        Adds flattened views: exec_chunk_flat, ref_chunk_flat, next_ref_flat.
        """
        out: dict[str, Tensor] = {}
        for k in (
            "state_vec",
            "exec_chunk",
            "ref_chunk",
            "reward_seq",
            "next_state_vec",
            "next_ref_chunk",
            "done",
            "actual_steps",
        ):
            if k not in batch:
                raise KeyError(f"ChunkACPolicy.forward missing batch key: {k!r}")
            v = batch[k]
            if not isinstance(v, Tensor):
                v = torch.as_tensor(v)
            out[k] = v
        out["exec_chunk_flat"] = out["exec_chunk"].flatten(start_dim=-2)
        out["ref_chunk_flat"] = out["ref_chunk"].flatten(start_dim=-2)
        out["next_ref_flat"] = out["next_ref_chunk"].flatten(start_dim=-2)
        return out

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        tx = self._coerce_batch(batch)

        c_loss = critic_loss(
            self.critic,
            self.target_critic,
            self.actor,
            tx,
            gamma=self.config.gamma,
            C=self.config.chunk_length,
            target_q_clip=self.config.target_q_clip,
        )
        soft_update(self.target_critic, self.critic, self.config.tau)
        self._critic_step += 1

        do_actor = (int(self._critic_step.item()) % self.config.actor_update_interval) == 0
        if do_actor:
            a_loss = self._actor_loss_without_critic_grads(tx)
            total = c_loss + a_loss
            info = {
                "loss": total.detach(),
                "loss_critic": c_loss.detach(),
                "loss_actor": a_loss.detach(),
                "critic_step": self._critic_step.detach().clone(),
            }
            return total, info

        info = {
            "loss": c_loss.detach(),
            "loss_critic": c_loss.detach(),
            "critic_step": self._critic_step.detach().clone(),
        }
        return c_loss, info

    def _actor_loss_without_critic_grads(self, tx: dict[str, Tensor]) -> Tensor:
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        for p in critic_params:
            p.requires_grad_(False)
        try:
            return actor_loss(self.actor, self.critic, tx, beta=self.config.beta)
        finally:
            for p in critic_params:
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Inference: predict_action_chunk + select_action
    # ------------------------------------------------------------------

    def _ensure_modifier(self) -> RLTActionModifier:
        if self.modifier is None:
            phase_ctrl = self._build_phase_controller()
            rl_token_module = self._rl_token_policy.rl_token
            self.modifier = RLTActionModifier(
                rl_token=rl_token_module,
                actor=self.actor,
                phase_ctrl=phase_ctrl,
                chunk_length=self.config.chunk_length,
                action_dim=self.config.action_dim,
                proprio_dim=self.config.proprio_dim,
                chunk_exec_steps=self.config.chunk_exec_steps,
                vla_ref=self.vla_ref,
            )
            self._prefix_capture = PrefixOutputCapture(
                token_pool_size=self.config.token_pool_size,
                image_only=self.config.image_only,
                num_image_tokens=self._compute_num_image_tokens(),
            )
            self._prefix_capture.attach(self._rl_token_policy._pi05)
        return self.modifier

    def _apply_phase_mode(self, ctrl: PhaseController) -> None:
        """Pin the phase controller to the phase encoded by phase_mode.

        always_rl -> critical, always_vla -> vla, manual -> left as-is. Must be
        re-applied after every reset(): modifier.reset() drops the controller
        back to VLA, which would otherwise silently disable always_rl from the
        second episode onward (record_loop resets the policy each episode).
        """
        if self.config.phase_mode == "always_rl":
            ctrl.trigger_critical()
        elif self.config.phase_mode == "always_vla":
            ctrl.trigger_vla()

    def _build_phase_controller(self) -> PhaseController:
        # ChunkACPolicyConfig.phase_mode (always_rl / always_vla / manual) is
        # validated in the config __post_init__. PhaseController only does
        # manual/learned transitions, so we pin the deploy phase explicitly.
        ctrl = PhaseController(mode="manual")
        self._apply_phase_mode(ctrl)
        return ctrl

    def _compute_num_image_tokens(self) -> int:
        pi05 = self._rl_token_policy._pi05
        pwx = pi05.model.paligemma_with_expert
        vision_cfg = pwx.paligemma.config.vision_config
        tokens_per_camera = (self.config.image_resolution[0] // vision_cfg.patch_size) ** 2
        return tokens_per_camera * len(self.config.camera_keys)

    def configure_rtc(
        self,
        rtc_config: RTCConfig,
        fps: float,
        action_queue_size_to_get_new_actions: int | None = None,
        vla_rtc_config: RTCConfig | None = None,
    ) -> None:
        """Enable RTC chunk replacement for deployment-time ``select_action``.

        ``predict_action_chunk`` already forwards RTC kwargs into the frozen pi0.5
        backbone. This runtime keeps an overlapping action queue so those kwargs
        contain the previous chunk leftovers and measured inference delay.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self._rtc_config = rtc_config
        self._vla_rtc_config = vla_rtc_config or rtc_config
        self._active_pi05_rtc_config = None
        self._set_pi05_rtc_config(rtc_config)
        self._rtc_action_queue = ActionQueue(rtc_config)
        self._rtc_latency_tracker = LatencyTracker()
        self._rtc_fps = float(fps)
        if action_queue_size_to_get_new_actions is None:
            action_queue_size_to_get_new_actions = max(1, self.config.chunk_length - 1)
        self._rtc_action_queue_size_to_get_new_actions = action_queue_size_to_get_new_actions
        self._rtc_worker = None
        self._rtc_worker_error = None
        self._rtc_generation = 0
        self._rtc_step_metadata.clear()
        self._rtc_selected_step_metadata.clear()

    @property
    def _rtc_runtime_enabled(self) -> bool:
        return self._rtc_config is not None

    def _rtc_config_for_phase(self, mod: RLTActionModifier) -> RTCConfig:
        if self._rtc_config is None:
            raise RuntimeError("RTC runtime is not configured")
        if not mod.is_rl_phase and self._vla_rtc_config is not None:
            return self._vla_rtc_config
        return self._rtc_config

    def _set_pi05_rtc_config(self, rtc_config: RTCConfig) -> None:
        if self._active_pi05_rtc_config is rtc_config:
            return
        pi05 = self._rl_token_policy._pi05
        pi05.config.rtc_config = rtc_config
        pi05.init_rtc_processor()
        self._active_pi05_rtc_config = rtc_config

    def _clone_rtc_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                cloned[key] = value.detach().clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _raise_rtc_worker_error(self) -> None:
        if self._rtc_worker_error is None:
            return
        error = self._rtc_worker_error
        self._rtc_worker_error = None
        raise error

    def _join_finished_rtc_worker(self) -> None:
        if self._rtc_worker is None or self._rtc_worker.is_alive():
            return
        self._rtc_worker.join()
        self._rtc_worker = None

    def _wait_for_rtc_worker(self) -> None:
        if self._rtc_worker is None:
            return
        self._rtc_worker.join()
        self._rtc_worker = None
        self._raise_rtc_worker_error()

    def _reset_rtc_runtime(self) -> None:
        with self._rtc_lock:
            self._rtc_generation += 1
            if self._rtc_action_queue is not None:
                self._rtc_action_queue = ActionQueue(self._rtc_action_queue.cfg)
            if self._rtc_latency_tracker is not None:
                self._rtc_latency_tracker.reset()
            self._rtc_step_metadata.clear()
            self._rtc_selected_step_metadata.clear()
            self._rtc_worker_error = None
        self._join_finished_rtc_worker()

    def _prepare_rtc_request(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Any], Tensor | None, int, int, float, int]:
        if self._rtc_action_queue is None or self._rtc_latency_tracker is None:
            raise RuntimeError("RTC runtime is not configured")

        time_per_chunk = 1.0 / self._rtc_fps
        with self._rtc_lock:
            prev_actions = self._rtc_action_queue.get_left_over()
            if prev_actions is not None:
                prev_actions = prev_actions.clone()
            action_index_before_inference = self._rtc_action_queue.get_action_index()
            generation = self._rtc_generation
        inference_delay = math.ceil(self._rtc_latency_tracker.max() / time_per_chunk)
        return (
            self._clone_rtc_batch(batch),
            prev_actions,
            inference_delay,
            action_index_before_inference,
            time.perf_counter(),
            generation,
        )

    def _predict_and_merge_rtc_chunk(
        self,
        batch: dict[str, Tensor],
        prev_actions: Tensor | None,
        inference_delay: int,
        action_index_before_inference: int,
        request_start_time: float,
        generation: int,
    ) -> None:
        if (
            self._rtc_action_queue is None
            or self._rtc_latency_tracker is None
            or self._rtc_config is None
        ):
            raise RuntimeError("RTC runtime is not configured")
        if generation != self._rtc_generation:
            return

        with self._rtc_inference_lock:
            if generation != self._rtc_generation:
                return
            mod = self._ensure_modifier()
            mod._step_metadata.clear()
            chunk = self.predict_action_chunk(
                batch,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )
            active_rtc_config = self._active_pi05_rtc_config or self._rtc_config
            if chunk.shape[0] != 1:
                raise ValueError(f"RTC deployment expects batch size 1, got {chunk.shape[0]}")

            original_actions = chunk.squeeze(0).detach()
            step_metadata = list(mod._step_metadata)
            mod._step_metadata.clear()
        if len(step_metadata) != len(original_actions):
            raise RuntimeError(
                f"RTC metadata/action length mismatch: {len(step_metadata)} != {len(original_actions)}"
            )

        new_latency = time.perf_counter() - request_start_time
        time_per_chunk = 1.0 / self._rtc_fps
        estimated_delay = math.ceil(new_latency / time_per_chunk)
        self._rtc_latency_tracker.add(new_latency)
        with self._rtc_lock:
            if generation != self._rtc_generation:
                return
            real_delay = max(0, self._rtc_action_queue.get_action_index() - action_index_before_inference)
            queue_before_merge = self._rtc_action_queue.qsize()
            self._rtc_action_queue.merge(
                original_actions,
                original_actions,
                real_delay,
                action_index_before_inference,
            )
            self._rtc_step_metadata = deque(step_metadata[real_delay:])

        log.info(
            "[RLT RTC] inference latency=%.1fms (estimated_delay=%d steps, real_delay=%d steps) | "
            "inference_delay used=%d | queue before merge=%d",
            new_latency * 1000.0,
            estimated_delay,
            real_delay,
            inference_delay,
            queue_before_merge,
        )
        min_refill_threshold = active_rtc_config.execution_horizon + estimated_delay
        if self._rtc_action_queue_size_to_get_new_actions < min_refill_threshold:
            log.warning(
                "[RLT RTC] action_queue_size_to_get_new_actions=%d is smaller than "
                "execution_horizon + delay (%d + %d). The queue may run dry under load.",
                self._rtc_action_queue_size_to_get_new_actions,
                active_rtc_config.execution_horizon,
                estimated_delay,
            )

    def _run_rtc_worker(
        self,
        batch: dict[str, Tensor],
        prev_actions: Tensor | None,
        inference_delay: int,
        action_index_before_inference: int,
        request_start_time: float,
        generation: int,
    ) -> None:
        try:
            self._predict_and_merge_rtc_chunk(
                batch,
                prev_actions,
                inference_delay,
                action_index_before_inference,
                request_start_time,
                generation,
            )
        except Exception as error:
            with self._rtc_lock:
                if generation == self._rtc_generation:
                    self._rtc_worker_error = error

    def _maybe_start_rtc_worker(self, batch: dict[str, Tensor]) -> None:
        if self._rtc_action_queue is None:
            raise RuntimeError("RTC runtime is not configured")
        if self._rtc_worker is not None and self._rtc_worker.is_alive():
            return
        self._join_finished_rtc_worker()
        self._raise_rtc_worker_error()
        if self._rtc_action_queue.qsize() > self._rtc_action_queue_size_to_get_new_actions:
            return
        request = self._prepare_rtc_request(batch)
        self._rtc_worker = Thread(target=self._run_rtc_worker, args=request, daemon=True)
        self._rtc_worker.start()

    def _select_action_rtc(self, batch: dict[str, Tensor]) -> Tensor:
        if self._rtc_action_queue is None:
            raise RuntimeError("RTC runtime is not configured")

        self._join_finished_rtc_worker()
        self._raise_rtc_worker_error()
        computed_sync = False
        if self._rtc_action_queue.empty():
            self._wait_for_rtc_worker()
            self._raise_rtc_worker_error()
        if self._rtc_action_queue.empty():
            self._predict_and_merge_rtc_chunk(*self._prepare_rtc_request(batch))
            computed_sync = True

        with self._rtc_lock:
            action = self._rtc_action_queue.get()
            step_metadata = self._rtc_step_metadata.popleft() if self._rtc_step_metadata else None
        if action is None:
            self._predict_and_merge_rtc_chunk(*self._prepare_rtc_request(batch))
            with self._rtc_lock:
                action = self._rtc_action_queue.get()
                step_metadata = self._rtc_step_metadata.popleft() if self._rtc_step_metadata else None
        if action is None:
            raise RuntimeError("RTC action queue is empty after refill")
        if step_metadata is None:
            raise RuntimeError("RTC metadata queue is empty after action pop")
        self._rtc_selected_step_metadata.append(step_metadata)

        if not computed_sync:
            self._maybe_start_rtc_worker(batch)
        return action.unsqueeze(0)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        mod = self._ensure_modifier()
        pi05 = self._rl_token_policy._pi05
        if self._rtc_config is not None:
            self._set_pi05_rtc_config(self._rtc_config_for_phase(mod))
        vla_chunk = pi05.predict_action_chunk(batch, **kwargs)
        vla_chunk = vla_chunk[:, :, : self.config.action_dim]
        prefix_tokens = self._prefix_capture.consume()
        proprio = batch["observation.state"][:, : self.config.proprio_dim]
        return mod.compute_chunk(vla_chunk, proprio, prefix_tokens)

    @torch.no_grad()
    def encode_replay_observation(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Encode an actual chunk boundary without altering execution queues."""
        self.eval()
        with self._rtc_inference_lock:
            mod = self._ensure_modifier()
            pi05 = self._rl_token_policy._pi05
            vla_chunk = pi05.predict_action_chunk(batch)
            prefix = self._prefix_capture.consume()
            proprio = batch["observation.state"][:, : self.config.proprio_dim]
            state = mod.encode_replay_state(proprio, prefix)
            ref = vla_chunk[:, : self.config.chunk_length, :self.config.action_dim]
        return state, ref

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        if self._rtc_runtime_enabled:
            return self._select_action_rtc(batch)

        mod = self._ensure_modifier()
        if mod.needs_new_chunk:
            chunk = self.predict_action_chunk(batch, **kwargs)
            mod.enqueue(chunk)
        return mod.pop_action()

    def reset(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        if self.modifier is not None:
            self.modifier.reset()
            # modifier.reset() resets the phase controller to VLA; re-pin the
            # configured deploy phase so always_rl survives episode resets.
            self._apply_phase_mode(self.modifier.phase_ctrl)

    def set_rl_mode(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        self._ensure_modifier().set_rl_mode()

    def set_vla_mode(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        self._ensure_modifier().set_vla_mode()

    def trigger_critical_phase(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        self._ensure_modifier().trigger_critical_phase()

    def interrupt_chunk(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        if self.modifier is not None:
            self.modifier.interrupt_chunk()

    def pop_step_metadata(self):
        if self._rtc_runtime_enabled:
            if len(self._rtc_selected_step_metadata) == 0:
                return None
            return self._rtc_selected_step_metadata.popleft()
        if self.modifier is None:
            return None
        return self.modifier.pop_step_metadata()

    def get_optim_params(self) -> list:
        return [
            {"params": list(self.actor.parameters())},
            {"params": list(self.critic.parameters())},
        ]

    # ------------------------------------------------------------------
    # Device + train-mode plumbing for the stashed backbone
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._rl_token_policy.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self._rl_token_policy.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self._rl_token_policy.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbones always in eval.
        self._rl_token_policy.eval()
        self.target_critic.eval()
        return self
