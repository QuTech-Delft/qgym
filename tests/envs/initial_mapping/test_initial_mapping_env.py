from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from stable_baselines3.common.env_checker import check_env

from qgym.envs import InitialMapping
from qgym.envs.initial_mapping import (
    BasicRewarder,
    EpisodeRewarder,
    InitialMappingState,
    SingleStepRewarder,
)
from qgym.templates import Rewarder


@pytest.fixture
def small_graph() -> nx.Graph:
    small_graph = nx.Graph()
    small_graph.add_edge(0, 1)
    return small_graph


@pytest.fixture
def small_env(small_graph: nx.Graph) -> InitialMapping:
    return InitialMapping(0.5, connection_graph=small_graph)


def test_validity(small_env: InitialMapping) -> None:
    check_env(
        small_env, warn=True
    )  # todo: maybe switch this to the gymnasium env checker
    assert True


@pytest.mark.parametrize(
    "render_mode,error_type",
    [(1, TypeError), ("test", ValueError)],
    ids=["TypeError", "ValueError"],
)
def test_unsupported_render_mode(
    small_graph: nx.Graph, render_mode: int | str, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        InitialMapping(0.5, connection_graph=small_graph, render_mode=render_mode)  # type: ignore[arg-type]


def test_init_custom_connection_graph(
    small_env: InitialMapping, small_graph: nx.Graph
) -> None:
    assert isinstance(small_env._state, InitialMappingState)
    assert nx.is_isomorphic(small_env._state.graphs["connection"]["graph"], small_graph)
    np.testing.assert_array_equal(
        small_env._state.graphs["connection"]["matrix"], np.array([[0, 1], [1, 0]])
    )


@pytest.mark.parametrize(
    "connection_graph_matrix",
    [np.array([[0, 1], [1, 0]]), [[0, 1], [1, 0]], csr_matrix([[0, 1], [1, 0]])],
)
def test_init_custom_connection_graph_matrix(
    small_graph: nx.Graph, connection_graph_matrix: ArrayLike
) -> None:
    env = InitialMapping(0.5, connection_graph_matrix=connection_graph_matrix)
    assert isinstance(env._state, InitialMappingState)
    assert nx.is_isomorphic(env._state.graphs["connection"]["graph"], small_graph)
    np.testing.assert_array_equal(
        env._state.graphs["connection"]["matrix"], np.array([[0, 1], [1, 0]])
    )


@pytest.mark.parametrize(
    "connection_grid_size",
    [(2, 1), [1, 2]],
)
def test_init_custom_connection_grid_size(
    small_graph: nx.Graph, connection_grid_size: list[int] | tuple[int, ...]
) -> None:
    env = InitialMapping(0.5, connection_grid_size=connection_grid_size)
    assert isinstance(env._state, InitialMappingState)
    assert nx.is_isomorphic(env._state.graphs["connection"]["graph"], small_graph)
    np.testing.assert_array_equal(
        env._state.graphs["connection"]["matrix"], np.array([[0, 1], [1, 0]])
    )


@pytest.mark.parametrize(
    "rewarder",
    [
        BasicRewarder(),
        EpisodeRewarder(),
        SingleStepRewarder(),
    ],
)
def test_init_custom_rewarder(rewarder: Rewarder) -> None:
    env = InitialMapping(1, connection_grid_size=(2, 2), rewarder=rewarder)
    assert env.rewarder == rewarder
    # Check that we made a copy for safety
    assert env.rewarder is not rewarder
