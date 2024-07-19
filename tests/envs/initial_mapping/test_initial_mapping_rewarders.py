from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Type, cast

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
from qgym.generators.graph import NullGraphGenerator
from qgym.templates.rewarder import Rewarder


def _episode_generator(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
) -> Iterator[tuple[InitialMappingState, NDArray[np.int_], InitialMappingState]]:
    connection_graph = nx.from_numpy_array(connection_graph_matrix)
    interaction_graph = nx.from_numpy_array(interaction_graph_matrix)
    new_state = InitialMappingState(connection_graph, NullGraphGenerator())
    new_state.reset(interaction_graph=interaction_graph)

    for i in range(connection_graph_matrix.shape[0]):
        action = np.array([i, i])
        old_state = deepcopy(new_state)
        new_state.update_state(action)
        yield old_state, action, new_state


@pytest.fixture(
    name="rewarder",
    params=(
        BasicRewarder(),
        SingleStepRewarder(),
        EpisodeRewarder(),
    ),
)
def rewarder_fixture(request: pytest.FixtureRequest) -> Rewarder:
    return cast(Rewarder, request.param)


@pytest.fixture(
    name="rewarder_class",
    params=(
        BasicRewarder,
        SingleStepRewarder,
        EpisodeRewarder,
    ),
)
def rewarder_class_fixture(request: pytest.FixtureRequest) -> type[Rewarder]:
    return cast(Type[Rewarder], request.param)


def test_illegal_actions(rewarder: Rewarder) -> None:
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


def test_inheritance(rewarder: Rewarder) -> None:
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
    rewarder_class: type[Rewarder],
    illegal_action_penalty: float,
    reward_per_edge: float,
    penalty_per_edge: float,
    reward_range: tuple[float, float],
) -> None:
    rewarder = rewarder_class(illegal_action_penalty, reward_per_edge, penalty_per_edge)  # type: ignore[call-arg]
    assert rewarder.reward_range == reward_range


def test_init(rewarder_class: type[Rewarder]) -> None:
    rewarder = rewarder_class(-3, 3, -2)  # type: ignore[call-arg]
    assert rewarder._reward_range == (-float("inf"), float("inf"))
    assert hasattr(rewarder, "_illegal_action_penalty")
    assert rewarder._illegal_action_penalty == -3
    assert hasattr(rewarder, "_reward_per_edge")
    assert rewarder._reward_per_edge == 3
    assert hasattr(rewarder, "_penalty_per_edge")
    assert rewarder._penalty_per_edge == -2


"""
Tests for the BasicRewarder
"""

empty_graph = np.zeros((3, 3), dtype=np.int_)
full_graph = np.array(([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), dtype=np.int_)


@pytest.mark.parametrize(
    "connection_graph_matrix, interaction_graph_matrix, rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, -1, -3]),
        (full_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 5, 15]),
    ],
)
def test_basic_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: list[float],
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
Tests for the SingleStepRewarder
"""


@pytest.mark.parametrize(
    "connection_graph_matrix,interaction_graph_matrix,rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, -1, -2]),
        (full_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 5, 10]),
    ],
)
def test_single_step_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: list[float],
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
Tests for the EpisodeRewarder
"""


@pytest.mark.parametrize(
    "connection_graph_matrix, interaction_graph_matrix, rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, 0, -3]),
        (full_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 0, 15]),
    ],
)
def test_episode_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: list[float],
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


"""
Tests for the Fidelity
"""

fidelity_graph = np.array(
    ([[0, 0.2, 0.6], [0.2, 0, 0.74], [0.6, 0.74, 0]]), dtype=np.float64
)


@pytest.mark.parametrize(
    "connection_graph_matrix, interaction_graph_matrix, rewards",
    [
        (empty_graph, empty_graph, [0, 0, 0]),
        (empty_graph, full_graph, [0, 0, -3]),
        (full_graph, empty_graph, [0, 0, 0]),
        (full_graph, full_graph, [0, 0, 15]),
        (fidelity_graph, empty_graph, [0, 0, 0]),
        (fidelity_graph, full_graph, [0, 0, 7.70]),
    ],
)
def test_fidelity_episode_rewarder(
    connection_graph_matrix: NDArray[np.int_],
    interaction_graph_matrix: NDArray[np.int_],
    rewards: list[float],
) -> None:
    episode_generator = _episode_generator(
        connection_graph_matrix, interaction_graph_matrix
    )

    rewarder = EpisodeRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        np.testing.assert_allclose(reward, rewards[i])
