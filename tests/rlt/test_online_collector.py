from __future__ import annotations

import pytest
import torch

from evo_rlt.adapters.lerobot.online_collector import RLTOnlineCollector
from evo_rlt.adapters.lerobot.record.annotations import (
    SOURCE_HUMAN,
    SOURCE_RL,
    SOURCE_VLA,
)
from evo_rlt.core.interfaces import (
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
)
from evo_rlt.core.replay_buffer import ReplayBuffer


def feed(collector, count, source=SOURCE_RL):
    for index in range(count):
        collector.on_frame(
            torch.tensor([float(index), float(index + 1)]),
            torch.tensor([float(index)]) if collector.needs_state else None,
            torch.full((3, 2), float(index)) if collector.needs_state else None,
            source,
            1.0,
        )


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 6, 7])
@pytest.mark.parametrize("success", [True, False])
def test_terminal_rewards_and_padding(steps, success):
    replay = ReplayBuffer()
    collector = RLTOnlineCollector(replay, 3, 2)
    collector.start_episode(4)
    feed(collector, steps)
    assert len(replay) == 0  # Outcome and terminal next state are not resolved yet.
    terminal = collector.flush_episode(success)
    assert len(replay) == (steps + 2) // 3
    assert terminal is replay.buffer[-1]
    actual = (steps - 1) % 3 + 1
    expected = torch.zeros(3)
    if success:
        expected[actual - 1] = 1
    assert torch.equal(terminal.reward_seq, expected)
    assert terminal.actual_steps.item() == actual
    assert terminal.exec_chunk[actual:].abs().sum().item() == 0
    for transition in list(replay.buffer)[:-1]:
        assert transition.done.item() == 0
        assert transition.reward_seq.sum().item() == 0
    assert terminal.done.item() == 1
    assert all(t.episode_id.item() == 4 for t in replay.buffer)
    assert all(
        t.source.item() == TRANSITION_SOURCE_RL_AUTONOMOUS for t in replay.buffer
    )
    batch = replay.sample(len(replay))
    assert batch["reward_seq"].sum().item() == float(success)


def test_next_state_is_current_boundary_and_episodes_do_not_link():
    replay = ReplayBuffer()
    collector = RLTOnlineCollector(replay, 3, 2)
    collector.start_episode(0)
    feed(collector, 7)
    collector.flush_episode(True)
    first, second, terminal = replay.buffer
    assert torch.equal(first.next_state_vec, torch.tensor([3.0]))
    assert torch.equal(second.next_state_vec, torch.tensor([6.0]))
    assert torch.equal(first.next_ref_chunk, second.ref_chunk)
    original_next_state = terminal.next_state_vec.clone()
    collector.start_episode(1)
    feed(collector, 1)
    collector.flush_episode(False)
    assert torch.equal(terminal.next_state_vec, original_next_state)
    assert replay.buffer[-1].reward_seq.sum() == 0


def test_human_override_is_not_lost_to_majority_vote_or_reference_replacement():
    replay = ReplayBuffer()
    collector = RLTOnlineCollector(replay, 3, 2)
    collector.start_episode(0)
    feed(collector, 2, SOURCE_VLA)
    collector.on_frame(torch.tensor([9.0, 10.0]), None, None, SOURCE_HUMAN, 1.0)
    collector.flush_episode(True)
    transition = replay.buffer[0]
    assert transition.intervention.item() == 1
    assert transition.source.item() == TRANSITION_SOURCE_HUMAN_OVERRIDE
    assert torch.equal(transition.ref_chunk, torch.zeros(3, 2))
    assert transition.exec_chunk[-1, 0].item() == 9


def test_discarded_and_empty_episodes_do_not_enter_replay():
    replay = ReplayBuffer()
    collector = RLTOnlineCollector(replay, 3, 2)
    collector.start_episode(0)
    feed(collector, 5)
    collector.discard_episode()
    collector.start_episode(0)
    assert collector.flush_episode(True) is None
    assert len(replay) == 0
    collector.start_episode(0)
    feed(collector, 3)
    collector.flush_episode(True)
    assert len(replay) == 1
    with pytest.raises(RuntimeError, match="No active episode"):
        collector.flush_episode(True)


def test_missing_boundary_state_raises_instead_of_reusing_old_state():
    collector = RLTOnlineCollector(ReplayBuffer(), 3, 2)
    collector.start_episode(0)
    feed(collector, 3)
    with pytest.raises(ValueError, match="current state_vec"):
        collector.on_frame(torch.ones(2), None, None, SOURCE_RL, 1.0)
