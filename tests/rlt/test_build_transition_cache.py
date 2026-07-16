from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset


def test_transition_cache_passes_video_backend(monkeypatch, tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache")

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_episodes = 0
            self.meta = SimpleNamespace(episodes=None)

    class FakeMetadata:
        fps = 20

        def __init__(self, **kwargs):
            pass

    class FakePolicy:
        config = SimpleNamespace(
            action_dim=12,
            chunk_size=50,
            image_only=False,
            proprio_dim=12,
            token_pool_size=0,
            vla_pretrained_path=None,
        )
        _num_image_tokens = 0
        _pi05 = object()
        rl_token = object()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def attach(self, pi05):
            pass

        def detach(self):
            pass

    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)

    def load_config(path, cli_overrides):
        captured["config_path"] = path
        captured["config_overrides"] = cli_overrides
        return FakePolicy.config

    monkeypatch.setattr(module.PreTrainedConfig, "from_pretrained", load_config)
    monkeypatch.setattr(
        module.RLTokenPolicy,
        "from_pretrained",
        lambda path, config: FakePolicy(),
    )
    monkeypatch.setattr(module, "PrefixOutputCapture", FakeCapture)
    monkeypatch.setattr(module, "make_rlt_token_pre_post_processors", lambda config: (object(), object()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_transition_cache.py",
            "--demo-dataset-repo-id",
            "local/demo",
            "--demo-dataset-root",
            "/tmp/demo",
            "--rl-token-policy-path",
            "/tmp/rl-token",
            "--norm-stats-path",
            "/tmp/norm-stats.pt",
            "--vla-pretrained-path",
            "/tmp/vla",
            "--tokenizer-path",
            "/tmp/tokenizer",
            "--output-dir",
            str(tmp_path),
            "--max-episodes",
            "0",
            "--video-backend",
            "video_reader",
        ],
    )

    module.main()

    assert captured["video_backend"] == "video_reader"
    assert captured["delta_timestamps"]["action"][1] == pytest.approx(0.05)
    assert captured["config_path"] == "/tmp/rl-token"
    assert captured["config_overrides"] == [
        "--vla_pretrained_path=/tmp/vla",
        "--tokenizer_path=/tmp/tokenizer",
        "--norm_stats_path=/tmp/norm-stats.pt",
    ]


class _FrameDataset(Dataset):
    def __init__(self, num_frames: int, chunk_length: int, action_dim: int):
        self._items = []
        for frame in range(num_frames):
            self._items.append(
                {
                    "observation.state": torch.tensor([frame, frame + 0.5]),
                    "action": torch.full((chunk_length, action_dim), float(frame)),
                }
            )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _Capture:
    prefix = None

    def consume(self):
        return self.prefix


class _Pi05:
    def __init__(self, capture):
        self._capture = capture

    def predict_action_chunk(self, batch):
        self._capture.prefix = batch["observation.state"][:, :1]
        return batch["action"] + 10.0


class _RLToken:
    def encode(self, prefix):
        return prefix


def test_encode_episode_builds_sparse_paper_aligned_transitions():
    from evo_rlt.cli.build_transition_cache import _encode_episode

    chunk_length = 3
    action_dim = 2
    capture = _Capture()
    transitions = _encode_episode(
        pi05=_Pi05(capture),
        rl_token=_RLToken(),
        preprocessor=lambda batch: batch,
        capture=capture,
        dataset=_FrameDataset(5, chunk_length, action_dim),
        frame_indices=[0, 1, 2, 3, 4],
        episode_last_frame=4,
        episode_success=True,
        chunk_length=chunk_length,
        stride=1,
        action_dim=action_dim,
        proprio_dim=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        empty_cache_every=100,
        task_str="test",
        ep_id=7,
    )

    assert len(transitions) == 2
    assert torch.equal(
        transitions[0].next_state_vec,
        torch.tensor([3.0, 3.0, 3.5]),
    )
    assert torch.equal(transitions[0].exec_chunk, torch.zeros(3, 2))
    assert torch.equal(transitions[0].ref_chunk, torch.full((3, 2), 10.0))
    assert torch.equal(transitions[0].reward_seq, torch.zeros(3))
    assert torch.equal(
        transitions[1].reward_seq,
        torch.tensor([0.0, 0.0, 1.0]),
    )
    assert transitions[1].done.item() == 1.0
