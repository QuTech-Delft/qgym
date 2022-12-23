from copy import deepcopy
from typing import Any, Dict, Iterator, List, Tuple

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from qgym.envs.initial_mapping.initial_mapping_rewarders import (
    BasicRewarder,
    EpisodeRewarder,
    SingleStepRewarder,
)
from qgym.envs.initial_mapping.initial_mapping_state import InitialMappingState
from qgym.templates.rewarder import Rewarder


def _episode_generator(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
) -> Iterator[Tuple[Dict[str, Any], NDArray[np.int_], Dict[str, Any]]]:
    connection_graph = nx.from_numpy_array(connection_graph_matrix)
    interaction_graph = nx.from_numpy_array(interaction_graph_matrix)
    new_state = InitialMappingState(connection_graph, 0)
    new_state.reset(interaction_graph=interaction_graph)

    for i in range(connection_graph_matrix.shape[0]):
        action = np.array([i, i])
        old_state = deepcopy(new_state)
        new_state.update_state(action)
        yield old_state, action, new_state


@pytest.fixture(
    name="rewarder", params=(BasicRewarder(), SingleStepRewarder(), EpisodeRewarder())
)
def _rewarder(request):
    return request.param


@pytest.fixture(
    name="rewarder_class", params=(BasicRewarder, SingleStepRewarder, EpisodeRewarder)
)
def _rewarder_class(request):
    return request.param


def test_illegal_actions(rewarder):
    episode_generator = _episode_generator(full_graph, empty_graph)
    _, _, new_state = next(episode_generator)
    old_state = deepcopy(new_state)

    reward = rewarder.compute_reward(
        old_state=old_state, action=np.array([0, 1]), new_state=new_state
    )
    assert reward == -100

    reward = rewarder.compute_reward(
        old_state=old_state, action=np.array([1, 0]), new_state=new_state
    )
    assert reward == -100


def test_inheritance(rewarder):
    assert isinstance(rewarder, Rewarder)


@pytest.mark.parametrize(
    "illegal_action_penalty,reward_per_edge,penalty_per_edge,reward_range",
    [
        (-1, 0, 0, (-float("inf"), 0)),
        (0, 1, 0, (0, float("inf"))),
        (0, 1, -1, (-float("inf"), float("inf"))),
    ],
)
def test_reward_range(
    rewarder_class,
    illegal_action_penalty,
    reward_per_edge,
    penalty_per_edge,
    reward_range,
):
    rewarder = rewarder_class(illegal_action_penalty, reward_per_edge, penalty_per_edge)
    assert rewarder.reward_range == reward_range


def test_init(rewarder_class):
    rewarder = rewarder_class(-3, 3, -2)
    assert rewarder._reward_range == (-float("inf"), float("inf"))
    assert rewarder._illegal_action_penalty == -3
    assert rewarder._reward_per_edge == 3
    assert rewarder._penalty_per_edge == -2


"""
Tests for the basic rewarder
"""

empty_graph = np.zeros((3, 3), dtype=np.int_)
full_graph = np.array(([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), dtype=np.int_)


@pytest.mark.parametrize(
    "connection_graph_matrix,interaction_graph_matrix,rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 5, 15]),
        (full_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, -1, -3]),
    ],
)
def test_basic_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: List[float],
) -> None:
    episode_generator = _episode_generator(
        connection_graph_matrix, interaction_graph_matrix
    )

    rewarder = BasicRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]


"""
Tests for the single step rewarder
"""


@pytest.mark.parametrize(
    "connection_graph_matrix,interaction_graph_matrix,rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 5, 10]),
        (full_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, -1, -2]),
    ],
)
def test_single_step_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: List[float],
) -> None:
    episode_generator = _episode_generator(
        connection_graph_matrix, interaction_graph_matrix
    )

    rewarder = SingleStepRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]


"""
Tests for the episode rewarder rewarder
"""


@pytest.mark.parametrize(
    "connection_graph_matrix,interaction_graph_matrix,rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 0, 15]),
        (full_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, 0, -3]),
    ],
)
def test_episode_step_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: List[float],
) -> None:
    episode_generator = _episode_generator(
        connection_graph_matrix, interaction_graph_matrix
    )

    rewarder = EpisodeRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]
