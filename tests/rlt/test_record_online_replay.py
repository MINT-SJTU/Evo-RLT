from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)

from evo_rlt.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset
from evo_rlt.adapters.lerobot.record.annotations import SOURCE_HUMAN
from evo_rlt.adapters.lerobot.record.online_replay import RecordingOnlineReplay


class ReplayPolicy:
    config = SimpleNamespace(device="cpu", use_amp=False, chunk_length=3, action_dim=2)

    def __init__(self):
        self.encoded_states = []

    def encode_replay_observation(self, batch):
        state = batch["observation.state"].clone()
        self.encoded_states.append(state)
        return state, torch.ones(1, 3, 2)


def make_preprocessor():
    return PolicyProcessorPipeline(
        steps=[
            RelativeActionsProcessorStep(enabled=True, action_names=["a", "b"]),
            NormalizerProcessorStep(
                features={
                    "observation.state": PolicyFeature(
                        type=FeatureType.STATE, shape=(2,)
                    ),
                    "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
                },
                norm_map={
                    FeatureType.STATE: NormalizationMode.MEAN_STD,
                    FeatureType.ACTION: NormalizationMode.MEAN_STD,
                },
                stats={
                    "observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)},
                    "action": {"mean": torch.zeros(2), "std": torch.full((2,), 2.0)},
                },
            ),
        ]
    )


def feed_recording(replay, count):
    for index in range(count):
        state = np.array([index, index + 1], dtype=np.float32)
        replay.on_frame(
            observation_frame={"observation.state": state},
            action=torch.from_numpy(state + 2),
            source_type=SOURCE_HUMAN,
            is_critical=1.0,
            task="insert",
            robot_type="test",
        )


def test_recording_normalizes_actions_encodes_boundaries_and_persists_outcome(tmp_path):
    policy = ReplayPolicy()
    replay = RecordingOnlineReplay(policy, make_preprocessor(), tmp_path)
    replay.start_episode(0)
    feed_recording(replay, 5)
    assert len(replay.buffer) == 0
    assert not (tmp_path / "chunk_transitions_train.pt").exists()
    assert len(policy.encoded_states) == 2
    assert torch.equal(policy.encoded_states[1], torch.tensor([[3.0, 4.0]]))
    replay.finish_episode("success")
    dataset = ChunkTransitionDataset(tmp_path)
    assert len(dataset) == 2
    assert torch.allclose(
        dataset[0]["exec_chunk"], torch.tensor([[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])
    )
    assert torch.allclose(
        dataset[1]["exec_chunk"][:2], torch.tensor([[1.0, 1.0], [1.5, 1.5]])
    )
    assert torch.equal(dataset[0]["next_state_vec"], dataset[1]["state_vec"])
    assert torch.equal(dataset[1]["reward_seq"], torch.tensor([0.0, 1.0, 0.0]))
    resumed = RecordingOnlineReplay(policy, make_preprocessor(), tmp_path)
    resumed.start_episode(1)
    feed_recording(resumed, 3)
    resumed.finish_episode("failure")
    dataset = ChunkTransitionDataset(tmp_path)
    assert len(dataset) == 3
    assert dataset[2]["reward_seq"].sum().item() == 0
    assert dataset[1]["reward_seq"].sum().item() == 1


def test_recording_requires_outcome_and_discards_rerecord(tmp_path):
    replay = RecordingOnlineReplay(ReplayPolicy(), make_preprocessor(), tmp_path)
    replay.start_episode(0)
    feed_recording(replay, 4)
    with pytest.raises(ValueError, match="Missing"):
        replay.finish_episode(None)
    assert len(replay.buffer) == 0
    replay.discard_episode()
    replay.start_episode(0)
    feed_recording(replay, 1)
    replay.finish_episode("success")
    assert len(ChunkTransitionDataset(tmp_path)) == 1


@pytest.mark.parametrize("skip_prefix", [False, True])
def test_robot_record_loop_passes_executed_actions_and_skips_prefix(
    tmp_path, monkeypatch, skip_prefix
):
    from evo_rlt.adapters.lerobot.record import loop

    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}

    class Robot:
        robot_type = "test"
        action_features = {"a": float, "b": float}
        frames = 0

        def get_observation(self):
            return {"a": float(self.frames), "b": float(self.frames + 1)}

        def send_action(self, action):
            # Emulate hardware clipping: the executed target differs from the
            # policy command and is what must enter the replay buffer.
            sent = {"a": float(self.frames + 2), "b": float(self.frames + 3)}
            self.frames += 1
            if self.frames == 5:
                events["exit_early"] = True
            return sent

    robot = Robot()

    class Policy(ReplayPolicy):
        config = SimpleNamespace(
            device="cpu", use_amp=False, chunk_length=3, action_dim=2, input_features={}
        )

        def reset(self):
            pass

        def set_rl_mode(self):
            pass

        def pop_step_metadata(self):
            return SimpleNamespace(phase=float(robot.frames > 1), source_type=1.0)

    class Dataset:
        fps = 30
        root = tmp_path
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["a", "b"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "complementary_info.policy_action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["a", "b"],
            },
        }
        episode_buffer = {"size": 0}

        def add_frame(self, frame):
            self.episode_buffer["size"] += 1

    class Identity:
        def reset(self):
            pass

    policy = Policy()
    replay = RecordingOnlineReplay(
        policy, make_preprocessor(), tmp_path / "online_replay"
    )
    replay.start_episode(0)
    monkeypatch.setattr(
        loop,
        "_predict_policy_action_with_acp_inference",
        lambda **kwargs: torch.full((2,), 100.0),
    )
    monkeypatch.setattr(loop, "precise_sleep", lambda duration: None)
    loop.record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=Dataset(),
        policy=policy,
        preprocessor=Identity(),
        postprocessor=Identity(),
        control_time_s=10,
        single_task="insert",
        rlt_online_collector=replay,
        skip_prefix_recording=skip_prefix,
    )
    replay.finish_episode("success")
    transitions = ChunkTransitionDataset(tmp_path / "online_replay")
    assert sum(t["actual_steps"].item() for t in transitions) == (
        4 if skip_prefix else 5
    )
    assert torch.allclose(
        transitions[0]["exec_chunk"], torch.tensor([[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])
    )
    assert sum(t["reward_seq"].sum().item() for t in transitions) == 1
