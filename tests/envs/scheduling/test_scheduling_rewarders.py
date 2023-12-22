from __future__ import annotations

from copy import deepcopy
from typing import Generator, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from qgym.custom_types import Gate
from qgym.envs.scheduling import (
    BasicRewarder,
    CommutationRulebook,
    EpisodeRewarder,
    MachineProperties,
)
from qgym.envs.scheduling.scheduling_state import SchedulingState
from qgym.templates import Rewarder


def _right_to_left_state_generator(
    circuit: list[Gate],
) -> Generator[tuple[SchedulingState, NDArray[np.int_], SchedulingState], None, None]:
    """
    Generate actions based on a circuit of timings. No illegal action will be taken.

    :param circuit: List of timings in cycle steps.
    """
    machine_properties = MachineProperties(2)
    machine_properties.add_gates({"x": 1, "y": 1, "measure": 5, "cnot": 2})
    rulebook = CommutationRulebook()
    new_state = SchedulingState(
        machine_properties=machine_properties,
        max_gates=10,
        dependency_depth=1,
        random_circuit_mode="workshop",
        rulebook=rulebook,
    )
    new_state.reset(circuit=circuit)

    for _ in range(len(circuit) * 5):
        legal_gates = np.nonzero(new_state.circuit_info.legal)[0]
        if len(legal_gates) > 0:
            action = np.array([legal_gates[-1], 0])
        else:
            action = np.array([0, 1])
        old_state = deepcopy(new_state)
        new_state.update_state(action)
        yield old_state, action, new_state
        if new_state.is_done():
            break


@pytest.fixture
def basic_rewarder() -> BasicRewarder:
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
def test_basic_rewarder_rewards(
    basic_rewarder: BasicRewarder, circuit: list[Gate], expected_reward: list[float]
) -> None:
    episode_generator = _right_to_left_state_generator(circuit)
    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = basic_rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )
        assert reward == expected_reward[i]


@pytest.fixture
def episode_rewarder() -> EpisodeRewarder:
    return EpisodeRewarder(-float("inf"), -1)


@pytest.mark.parametrize(
    "circuit,expected_reward",
    [
        ([Gate("x", 1, 1)], [-1]),
        ([Gate("x", 1, 1), Gate("x", 1, 1)], [0, 0, -2]),
        ([Gate("x", 1, 1), Gate("x", 1, 1), Gate("measure", 1, 1)], [0] * 8 + [-7]),
    ],
)
def test_episode_rewarder_rewards(
    episode_rewarder: EpisodeRewarder, circuit: list[Gate], expected_reward: list[float]
) -> None:
    episode_generator = _right_to_left_state_generator(circuit)
    for i, (old_state, action, new_state) in enumerate(episode_generator):
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
    illegal_action_p: float,
    update_cycle_p: float,
    schedule_gate_b: float,
    reward_range: tuple[float, float],
) -> None:
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
def test_reward_range_episode_rewarder(
    illegal_action_p: float, cycle_used_p: float, reward_range: tuple[float, float]
) -> None:
    rewarder = EpisodeRewarder(illegal_action_p, cycle_used_p)
    assert rewarder.reward_range == reward_range


@pytest.fixture(name="rewarder", params=(BasicRewarder(), EpisodeRewarder()))
def _rewarder(request: pytest.FixtureRequest) -> Rewarder:
    return cast(Rewarder, request.param)


def test_illegal_actions(rewarder: Rewarder) -> None:
    circuit = [Gate("x", 1, 1), Gate("y", 1, 1)]
    old_state, _, _ = next(_right_to_left_state_generator(circuit))
    assert hasattr(rewarder, "_is_illegal")
    assert rewarder._is_illegal([0, 0], old_state)
    assert not rewarder._is_illegal([1, 0], old_state)
