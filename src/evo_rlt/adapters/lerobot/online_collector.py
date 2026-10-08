from __future__ import annotations

import torch

from evo_rlt.adapters.lerobot.record.annotations import (
    SOURCE_HUMAN,
    SOURCE_RL,
    SOURCE_VLA,
)
from evo_rlt.core.interfaces import (
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
    ChunkTransition,
)
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.rewards import build_reward_seq


class RLTOnlineCollector:
    """Stage normalized robot chunks until the episode outcome is confirmed.

    The caller supplies a freshly encoded observation/reference at each chunk
    boundary. Only finalized episodes enter replay, so unresolved next states,
    missing terminal rewards and discarded rerecords cannot be sampled.
    """

    def __init__(self, replay_buffer: ReplayBuffer, chunk_length: int, action_dim: int):
        if chunk_length <= 0 or action_dim <= 0:
            raise ValueError("chunk_length and action_dim must be positive")
        self._buffer = replay_buffer
        self._C = chunk_length
        self._action_dim = action_dim
        self._active = False
        self._episode_id = -1
        self._frame_actions: list[torch.Tensor] = []
        self._frame_sources: list[float] = []
        self._transitions: list[ChunkTransition] = []
        self._chunk_state: torch.Tensor | None = None
        self._chunk_ref: torch.Tensor | None = None
        self._chunk_is_critical = 0.0

    @property
    def needs_state(self) -> bool:
        return not self._frame_actions

    def start_episode(self, episode_id: int) -> None:
        if self._active:
            raise RuntimeError(
                "Finalize or discard the current episode before starting another"
            )
        self._episode_id = episode_id
        self._active = True

    def discard_episode(self) -> None:
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._transitions.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        self._active = False

    def on_frame(
        self,
        action: torch.Tensor,
        state_vec: torch.Tensor | None,
        ref_chunk: torch.Tensor | None,
        source_type: float,
        is_critical: float,
    ) -> ChunkTransition | None:
        if not self._active:
            raise RuntimeError("start_episode must be called before on_frame")
        if action.shape != (self._action_dim,):
            raise ValueError(
                f"Expected action shape {(self._action_dim,)}, got {tuple(action.shape)}"
            )
        if source_type not in (SOURCE_VLA, SOURCE_RL, SOURCE_HUMAN):
            raise ValueError(f"Unknown recording source: {source_type}")
        if self.needs_state:
            if state_vec is None or ref_chunk is None:
                raise ValueError(
                    "Every chunk requires a current state_vec and ref_chunk"
                )
            if state_vec.ndim != 1 or ref_chunk.shape != (self._C, self._action_dim):
                raise ValueError(
                    "Expected an unbatched state and a (chunk_length, action_dim) reference"
                )
            self._chunk_state = state_vec.detach().cpu().clone()
            self._chunk_ref = ref_chunk.detach().cpu().clone()
            if self._transitions:
                self._transitions[-1].next_state_vec = self._chunk_state
                self._transitions[-1].next_ref_chunk = self._chunk_ref
        self._chunk_is_critical = max(self._chunk_is_critical, float(is_critical))
        self._frame_actions.append(action.detach().cpu().clone())
        self._frame_sources.append(source_type)
        if len(self._frame_actions) == self._C:
            return self._emit_transition()
        return None

    def flush_episode(self, episode_success: bool) -> ChunkTransition | None:
        if not self._active:
            raise RuntimeError("No active episode to finalize")
        if not isinstance(episode_success, bool):
            raise TypeError("episode_success must be a bool")
        if self._frame_actions:
            self._emit_transition()
        terminal = self._transitions[-1] if self._transitions else None
        if terminal is not None:
            terminal.done = torch.tensor(1.0)
            terminal.reward_seq = build_reward_seq(
                self._C,
                is_terminal_chunk=True,
                episode_success=episode_success,
                actual_steps=terminal.actual_steps,
            )
            for transition in self._transitions:
                self._buffer.add(transition)
        self.discard_episode()
        return terminal

    def _emit_transition(self) -> ChunkTransition:
        actual = len(self._frame_actions)
        exec_chunk = torch.zeros(
            self._C, self._action_dim, dtype=self._frame_actions[0].dtype
        )
        exec_chunk[:actual] = torch.stack(self._frame_actions)
        # Any human override identifies the chunk as intervention data, even
        # when the human supplies fewer than half of its executed actions.
        if SOURCE_HUMAN in self._frame_sources:
            source = TRANSITION_SOURCE_HUMAN_OVERRIDE
        elif SOURCE_RL in self._frame_sources:
            source = TRANSITION_SOURCE_RL_AUTONOMOUS
        else:
            source = TRANSITION_SOURCE_WARMUP_VLA
        transition = ChunkTransition(
            state_vec=self._chunk_state,
            exec_chunk=exec_chunk,
            ref_chunk=self._chunk_ref,
            reward_seq=build_reward_seq(self._C, False, False, actual_steps=actual),
            next_state_vec=self._chunk_state,
            next_ref_chunk=self._chunk_ref,
            done=torch.tensor(0.0),
            intervention=torch.tensor(float(SOURCE_HUMAN in self._frame_sources)),
            actual_steps=torch.tensor(actual),
            source=torch.tensor(source),
            episode_id=torch.tensor(self._episode_id),
            is_critical=torch.tensor(self._chunk_is_critical),
        )
        self._transitions.append(transition)
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_is_critical = 0.0
        return transition
