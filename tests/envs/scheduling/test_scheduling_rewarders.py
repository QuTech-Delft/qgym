from typing import Iterator, List

import numpy as np
import pytest
from qgym.custom_types import Gate
from qgym.envs.scheduling import BasicRewarder, EpisodeRewarder


def _right_to_left_action_generator(circuit: List[Gate]) -> Iterator[List[int]]:
    """
    Generate actions based on a circuit of timings. No illegal action will be taken.

    :param circuit: List of timings in cycle steps.
    """
    gate_cycles = {"x": 1, "measure": 5}
    for i in range(len(circuit) - 1, 0, -1):
        yield np.array([i, 0])
        gate = circuit[i]
        for _ in range(gate_cycles[gate.name]):
            yield np.array([i, 1])

    yield np.array([0, 0])


@pytest.fixture
def basic_rewarder():
    return BasicRewarder(-float("inf"), -1, 100)


@pytest.mark.parametrize(
    "circuit,expected_reward",
    [
        ([Gate("x", 1, 1)], [100]),
        ([Gate("x", 1, 1), Gate("x", 1, 1)], [100, -1, 100]),
        (
            [Gate("x", 1, 1), Gate("x", 1, 1), Gate("measure", 1, 1)],
            [100, -1, -1, -1, -1, -1, 100, -1, 100],
        ),
    ],
)
def test_basic_rewarder_rewards(basic_rewarder, circuit, expected_reward):
    action_generator = _right_to_left_action_generator(circuit)
    old_state = {"legal_actions": [True] * len(circuit)}
    new_state = {}
    for (i, action) in enumerate(action_generator):
        reward = basic_rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )
        assert reward == expected_reward[i]


@pytest.fixture
def episode_rewarder():
    return EpisodeRewarder(-float("inf"), -1)


@pytest.mark.parametrize(
    "circuit,expected_reward",
    [
        ([Gate("x", 1, 1)], [-2]),
        ([Gate("x", 1, 1), Gate("x", 1, 1)], [0, 0, -2]),
        (
            [Gate("x", 1, 1), Gate("x", 1, 1), Gate("measure", 1, 1)],
            [0, 0, 0, 0, 0, 0, 0, 0, -6],
        ),
    ],
)
def test_episode_rewarder_rewards(episode_rewarder, circuit, expected_reward):
    action_generator = _right_to_left_action_generator(circuit)
    old_state = {
        "legal_actions": [True] * len(circuit),
    }
    new_state = {"schedule": np.array([-1])}
    for (i, action) in enumerate(action_generator):
        if action[0] == 0:
            new_state = {
                "schedule": np.ones(len(circuit)),
                "encoded_circuit": circuit,
                "gate_cycle_length": {"x": 1, "measure": 5},
            }
        reward = episode_rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )
        assert reward == expected_reward[i]


@pytest.mark.parametrize(
    "illegal_action_p,update_cycle_p,schedule_gate_b,reward_range",
    [
        (-1, 0, 0, (-float("inf"), 0)),
        (0, 0, 1, (0, float("inf"))),
        (0, -1, 1, (-float("inf"), float("inf"))),
    ],
)
def test_reward_range_basic_rewarder(
    illegal_action_p, update_cycle_p, schedule_gate_b, reward_range
):
    rewarder = BasicRewarder(illegal_action_p, update_cycle_p, schedule_gate_b)
    assert rewarder.reward_range == reward_range


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "illegal_action_p,cycle_used_p,reward_range",
    [
        (-1, 0, (-float("inf"), 0)),
        (0, 1, (0, float("inf"))),
        (1, -1, (-float("inf"), float("inf"))),
    ],
)
def test_reward_range_episode_rewarder(illegal_action_p, cycle_used_p, reward_range):
    rewarder = EpisodeRewarder(illegal_action_p, cycle_used_p)
    assert rewarder.reward_range == reward_range


@pytest.fixture(name="rewarder", params=(BasicRewarder(), EpisodeRewarder()))
def _rewarder(request):
    return request.param


def test_illegal_actions(rewarder):
    old_state = {"legal_actions": [False, True]}

    assert rewarder._is_illegal([0, 0], old_state)
    assert not rewarder._is_illegal([1, 0], old_state)
