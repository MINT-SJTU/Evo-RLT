from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from lerobot.policies.utils import prepare_observation_for_inference

from evo_rlt.adapters.lerobot.offline_dataset import (
    load_transition_cache,
    save_transition_cache,
)
from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
from evo_rlt.adapters.lerobot.record.annotations import (
    EPISODE_SUCCESS,
    resolve_episode_success_label,
)
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.rl_token import RLTokenModule
from evo_rlt.core.utils import filter_encoder_only


def _encoder_fingerprint(rl_token: RLTokenModule) -> str:
    """Identify the encoder that produced the stored RL-token features."""
    digest = hashlib.sha256()
    num_heads = rl_token.encoder.layers[0].self_attn.num_heads
    digest.update(f"{num_heads}\n".encode())
    weights, _ = filter_encoder_only(rl_token.state_dict())
    for name, tensor in sorted(weights.items()):
        digest.update(json.dumps((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        bytes_view = tensor.detach().cpu().contiguous().view(torch.uint8)
        digest.update(memoryview(bytes_view.numpy()))
    return digest.hexdigest()


class RecordingOnlineReplay:
    """Bridge robot frames and confirmed outcomes to a persistent training cache."""

    def __init__(self, policy, preprocessor, cache_dir: str | Path):
        self.policy = policy
        self.preprocessor = preprocessor
        self.cache_dir = Path(cache_dir)
        self._chunk_observation: dict | None = None
        self._cache_path = self.cache_dir / "chunk_transitions_train.pt"
        self._fingerprint_path = self.cache_dir / "rl_token_encoder.sha256"
        self._fingerprint = _encoder_fingerprint(policy.rl_token)
        self._validate_encoder()
        self.buffer = (
            load_transition_cache(self.cache_dir, "train")
            if self._cache_path.exists()
            else ReplayBuffer()
        )
        self.collector = RLTOnlineCollector(
            self.buffer, policy.config.chunk_length, policy.config.action_dim
        )

    def _validate_encoder(self, *, initialize: bool = False) -> None:
        if self._fingerprint_path.exists():
            if self._fingerprint_path.read_text().strip() != self._fingerprint:
                raise ValueError(
                    f"RL-token encoder fingerprint mismatch for {self.cache_dir}. "
                    "Use the original encoder or start a new online replay cache."
                )
        elif self._cache_path.exists():
            raise ValueError(
                f"Online replay cache {self.cache_dir} has no RL-token encoder fingerprint. "
                "Start a new online replay cache."
            )
        elif initialize:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._fingerprint_path.open("x") as fingerprint_file:
                fingerprint_file.write(f"{self._fingerprint}\n")

    def start_episode(self, episode_id: int) -> None:
        self.collector.start_episode(episode_id)
        self._chunk_observation = None

    def discard_episode(self) -> None:
        self.collector.discard_episode()
        self._chunk_observation = None

    @torch.inference_mode()
    def on_frame(
        self, *, observation_frame, action, source_type, is_critical, task, robot_type
    ) -> None:
        device = torch.device(self.policy.config.device)
        if self.collector.needs_state:
            self._chunk_observation = {
                key: value.copy() for key, value in observation_frame.items()
            }
        # Relative actions in an offline action chunk are all measured against
        # its anchor state, rather than each action's own per-frame state.
        batch = prepare_observation_for_inference(
            dict(self._chunk_observation), device, task, robot_type
        )
        # Feed executed robot actions through the same SFT preprocessing as
        # offline cache building, including relative-action and quantile steps.
        batch["action"] = action.reshape(1, 1, -1).to(device)
        with (
            torch.autocast(device.type)
            if device.type == "cuda" and self.policy.config.use_amp
            else nullcontext()
        ):
            batch = self.preprocessor(batch)
            state, ref = (None, None)
            if self.collector.needs_state:
                state_batch, ref_batch = self.policy.encode_replay_observation(batch)
                if state_batch.shape[0] != 1 or ref_batch.shape[0] != 1:
                    raise ValueError("Real-robot online replay expects batch size 1")
                state, ref = state_batch[0], ref_batch[0]
        self.collector.on_frame(
            action=batch["action"][0, 0, : self.policy.config.action_dim],
            state_vec=state,
            ref_chunk=ref,
            source_type=source_type,
            is_critical=is_critical,
        )

    def finish_episode(self, episode_success: str | None) -> None:
        label = resolve_episode_success_label(
            explicit_label=episode_success, require_label=True
        )
        self._validate_encoder(initialize=True)
        self.collector.flush_episode(label == EPISODE_SUCCESS)
        self._chunk_observation = None
        save_transition_cache(list(self.buffer.buffer), self.cache_dir, "train")
