from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset


@pytest.mark.parametrize(
    ("active_camera_indices", "expected_token"),
    [([1], 2.5), (None, 1.5)],
)
def test_cache_cli_uses_checkpoint_camera_selection(
    monkeypatch, tmp_path, active_camera_indices, expected_token
):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache")
    from evo_rlt.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset

    captured = {}

    def load_dataset(**kwargs):
        captured.update(kwargs)
        dataset = _FrameDataset(num_frames=3, chunk_length=2, action_dim=2)
        dataset.num_episodes = 1
        dataset.meta = SimpleNamespace(episodes={
            "dataset_from_index": [0],
            "dataset_to_index": [3],
            "episode_success": ["success"],
        })
        return dataset

    class FakeMetadata:
        fps = 20

        def __init__(self, **kwargs):
            pass

    # Two tokens per camera, followed by a language token.
    prefix = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]).view(1, 5, 1)

    def forward(batch_size):
        return [prefix.expand(batch_size, -1, -1), None], None

    backbone = SimpleNamespace(forward=forward)

    def predict_action_chunk(batch):
        backbone.forward(batch["action"].shape[0])
        return batch["action"]

    class FakePolicy:
        config = SimpleNamespace(
            action_dim=2,
            image_only=True,
            proprio_dim=2,
            token_pool_size=1,
            active_camera_indices=active_camera_indices,
            num_per_camera=2,
            vla_pretrained_path=None,
        )
        _num_image_tokens = 4
        _pi05 = SimpleNamespace(
            model=SimpleNamespace(paligemma_with_expert=backbone),
            predict_action_chunk=predict_action_chunk,
        )
        rl_token = _RLToken()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(module, "LeRobotDataset", load_dataset)
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
    monkeypatch.setattr(module, "make_rlt_token_pre_post_processors", lambda config: (lambda batch: batch, None))
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
            "--chunk-length",
            "2",
            "--num-workers",
            "0",
            "--train-ratio",
            "1",
            "--device",
            "cpu",
            "--video-backend",
            "video_reader",
        ],
    )

    module.main()

    cache = ChunkTransitionDataset(tmp_path)
    assert len(cache) == 1
    assert cache[0]["state_vec"][0].item() == expected_token
    assert cache[0]["next_state_vec"][0].item() == expected_token
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
