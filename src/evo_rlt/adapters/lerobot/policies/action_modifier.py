from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.phase_controller import PhaseController
from evo_rlt.core.rl_token import RLTokenModule
from evo_rlt.core.utils import flatten_chunk, postprocess_prefix_tokens, unflatten_chunk


class PrefixOutputCapture:
    """Capture prefix hidden states from PI05Policy's PaliGemmaWithExpertModel.

    PI05's ``sample_actions`` calls ``paligemma_with_expert.forward()``
    directly (not via ``__call__``), so standard PyTorch forward hooks
    never fire.  Instead we monkey-patch ``forward`` to intercept the
    prefix-only call (``inputs_embeds=[prefix_embs, None]``).

    After capture the raw prefix tokens are optionally sliced to image-only
    (dropping language tokens) and pooled from (B, ~968, 2048) to
    (B, token_pool_size, 2048) via adaptive average pooling.
    """

    def __init__(
        self,
        token_pool_size: int = 64,
        image_only: bool = False,
        num_image_tokens: int = 0,
    ):
        self.token_pool_size = token_pool_size
        self.image_only = image_only
        self.num_image_tokens = num_image_tokens
        self._captured: Tensor | None = None
        self._original_forward = None
        self._target = None

    def attach(self, policy) -> None:
        """Monkey-patch ``forward`` on ``policy.model.paligemma_with_expert``.

        When image_only is true and num_image_tokens is unset (0), derive it from
        the SigLIP vision config so callers don't need to know paligemma internals.
        """
        target = policy.model.paligemma_with_expert
        self._target = target
        self._original_forward = target.forward

        if self.image_only and self.num_image_tokens == 0:
            self.num_image_tokens = self._infer_num_image_tokens(policy)

        capture = self  # closure reference

        def patched_forward(*args, **kwargs):
            result = capture._original_forward(*args, **kwargs)
            outputs, _past_kv = result
            prefix_tokens = outputs[0]
            if prefix_tokens is not None:
                capture._captured = postprocess_prefix_tokens(
                    prefix_tokens.detach().float(),
                    image_only=capture.image_only,
                    num_image_tokens=capture.num_image_tokens,
                    pool_size=capture.token_pool_size,
                )
            return result

        target.forward = patched_forward

    @staticmethod
    def _infer_num_image_tokens(policy) -> int:
        pi05_cfg = policy.config
        vision_cfg = policy.model.paligemma_with_expert.paligemma.config.vision_config
        n_per_cam = (pi05_cfg.image_resolution[0] // vision_cfg.patch_size) ** 2
        return n_per_cam * len(pi05_cfg.image_features)

    def consume(self) -> Tensor:
        """Return and clear the captured prefix tokens.

        Raises AssertionError if no prefix output has been captured yet (i.e.
        the VLA forward pass has not run since the last consume).
        """
        assert self._captured is not None, (
            "No prefix_output captured -- VLA forward not yet called"
        )
        result = self._captured
        self._captured = None
        return result

    def detach(self) -> None:
        """Restore the original forward method."""
        if self._original_forward is not None and self._target is not None:
            self._target.forward = self._original_forward
            self._original_forward = None
            self._target = None


@dataclass
class RLTStepMetadata:
    """Per-step metadata emitted alongside each popped action."""

    phase: float  # 0.0 = VLA, 1.0 = critical/RL
    source_type: float  # 0.0 = VLA action, 1.0 = RL action


class RLTActionModifier(nn.Module):
    """RL Token Encoder + Actor + Phase Controller + Chunk Queue.

    Sits between the VLA action output and the final action used by the robot.
    In VLA phase the VLA chunk is passed through unchanged; in RL phase the
    Actor refines the chunk conditioned on the RL-token state representation.
    """

    def __init__(
        self,
        rl_token: RLTokenModule,
        actor: ChunkActor,
        phase_ctrl: PhaseController,
        chunk_length: int,
        action_dim: int,
        proprio_dim: int,
        chunk_exec_steps: int = 25,
        vla_ref: bool = True,
    ):
        super().__init__()
        self.rl_token = rl_token
        self.actor = actor
        self.phase_ctrl = phase_ctrl
        self.chunk_length = chunk_length
        self.chunk_exec_steps = chunk_exec_steps
        self.vla_ref = vla_ref
        self._cc_log_count = 0  # throttle for the compute_chunk ref diagnostic
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self._action_queue: deque[Tensor] = deque()
        self._step_metadata: deque[RLTStepMetadata] = deque()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_rl_phase(self) -> bool:
        return self.phase_ctrl.is_critical

    @property
    def needs_new_chunk(self) -> bool:
        return len(self._action_queue) == 0

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def encode_replay_state(self, proprio: Tensor, prefix_tokens: Tensor) -> Tensor:
        """Encode a current observation in the same space used by the actor."""
        return torch.cat([self.rl_token.encode(prefix_tokens), proprio], dim=-1)

    @torch.no_grad()
    def compute_chunk(
        self,
        vla_chunk: Tensor,
        proprio: Tensor,
        prefix_tokens: Tensor,
    ) -> Tensor:
        """Compute action chunk, either VLA pass-through or RL-refined.

        Args:
            vla_chunk: (B, H, action_dim) normalised VLA action chunk.
            proprio: (B, proprio_dim) normalised proprioceptive state.
            prefix_tokens: (B, pool_size, token_dim) pooled VLA prefix tokens.

        Returns:
            VLA phase:  (B, chunk_exec_steps, action_dim)
            RL phase:   (B, chunk_length, action_dim) in [-1, 1].
        """
        phase_val = 1.0 if self.is_rl_phase else 0.0
        source_val = phase_val

        if not self.is_rl_phase:
            # VLA phase: take the first chunk_exec_steps consecutive actions.
            n = min(self.chunk_exec_steps, vla_chunk.shape[1])
            self._enqueue_metadata(phase_val, source_val, n)
            return vla_chunk[:, :n, :]

        # RL phase: take first chunk_length frames as actor reference
        ref_chunk = vla_chunk[:, :self.chunk_length, :]
        state_vec = self.encode_replay_state(proprio, prefix_tokens)
        ref_flat = flatten_chunk(ref_chunk)
        if not self.vla_ref:
            # Hide the VLA reference from the actor: training ref-dropout
            # multiplies ref_chunk_flat by a 0/1 mask, so a dropped sample
            # is exactly an all-zero ref. Reproduce that here.
            ref_flat = torch.zeros_like(ref_flat)
        should_log = self._cc_log_count < 3 or self._cc_log_count % 30 == 0
        mu, _ = self.actor(state_vec, ref_flat, training=False)
        chunk = unflatten_chunk(mu, self.chunk_length).clamp(-1, 1)
        if should_log:
            delta = (chunk - ref_chunk).abs()
            # Diagnostic: the VLA ref is only the actor input; the returned
            # chunk below is the RLT actor output that will be executed.
            print(
                f"[RLT source=RLT_ACTOR vla_ref={self.vla_ref}] "
                f"compute_chunk #{self._cc_log_count}: "
                f"vla_ref[0,0,:4]={[round(v, 4) for v in ref_chunk[0, 0, :4].tolist()]}, "
                f"actor_out[0,0,:4]={[round(v, 4) for v in chunk[0, 0, :4].tolist()]}, "
                f"mean_abs_delta={delta.mean().item():.4f}, "
                f"max_abs_delta={delta.max().item():.4f}, "
                f"ref-into-actor abs-sum={ref_flat.abs().sum().item():.4f}",
                flush=True,
            )
        self._cc_log_count += 1

        self._enqueue_metadata(phase_val, source_val, self.chunk_length)
        return chunk

    def _enqueue_metadata(self, phase: float, source: float, count: int) -> None:
        """Enqueue metadata entries for every step in the upcoming chunk."""
        for _ in range(count):
            self._step_metadata.append(RLTStepMetadata(phase=phase, source_type=source))

    def enqueue(self, chunk: Tensor) -> None:
        """Enqueue chunk steps into the action queue.

        Args:
            chunk: (B, C, action_dim).
        """
        self._action_queue.extend(chunk.transpose(0, 1))

    def pop_action(self) -> Tensor:
        """Pop and return the next single-step action from the queue."""
        return self._action_queue.popleft()

    def pop_step_metadata(self) -> RLTStepMetadata | None:
        """Pop and return the next step's metadata, or None if empty."""
        if len(self._step_metadata) == 0:
            return None
        return self._step_metadata.popleft()

    # ------------------------------------------------------------------
    # Phase control (duck-typed interface for recording_loop)
    # ------------------------------------------------------------------

    def set_rl_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_critical()

    def set_vla_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_vla()

    def trigger_critical_phase(self) -> None:
        """Toggle between VLA and critical phase, clearing queues."""
        self.interrupt_chunk()
        if self.phase_ctrl.is_critical:
            self.phase_ctrl.trigger_vla()
        else:
            self.phase_ctrl.trigger_critical()

    def interrupt_chunk(self) -> None:
        """Clear both action and metadata queues (e.g. on phase switch)."""
        self._action_queue.clear()
        self._step_metadata.clear()

    def reset(self) -> None:
        """Reset queues and phase controller to initial state."""
        self._action_queue.clear()
        self._step_metadata.clear()
        self.phase_ctrl.reset()
