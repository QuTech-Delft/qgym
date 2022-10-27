import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from stable_baselines3.common.env_checker import check_env

from qgym.envs import InitialMapping
from qgym.envs.initial_mapping.initial_mapping_rewarders import (
    BasicRewarder,
    EpisodeRewarder,
    SingleStepRewarder,
)


@pytest.fixture
def small_graph():
    small_graph = nx.Graph()
    small_graph.add_edge(0, 1)
    return small_graph


@pytest.fixture
def small_env(small_graph):
    return InitialMapping(0.5, connection_graph=small_graph)


def test_validity(small_env) -> None:
    check_env(small_env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True


@pytest.mark.parametrize(
    "mode,error_type,error_msg",
    [
        (1, TypeError, "'mode' must be a string, but was of type <class 'int'>"),
        ("test", ValueError, "The given render mode is not supported"),
    ],
)
def test_unsupported_render_mode(small_env, mode, error_type, error_msg):
    with pytest.raises(error_type, match=error_msg):
        small_env.render(mode=mode)
    small_env.close()


def test_init_custom_connection_graph(small_env, small_graph):
    assert nx.is_isomorphic(small_env._connection_graph, small_graph)
    assert (
        small_env._state["connection_graph_matrix"] == np.array([[0, 1], [1, 0]])
    ).all()


@pytest.mark.parametrize(
    "connection_graph_matrix",
    [np.array([[0, 1], [1, 0]]), [[0, 1], [1, 0]], csr_matrix([[0, 1], [1, 0]])],
)
def test_init_custom_connection_graph_matrix(small_graph, connection_graph_matrix):
    env = InitialMapping(0.5, connection_graph_matrix=connection_graph_matrix)
    assert nx.is_isomorphic(env._connection_graph, small_graph)
    assert (env._state["connection_graph_matrix"] == np.array([[0, 1], [1, 0]])).all()


@pytest.mark.parametrize(
    "connection_grid_size",
    [(2, 1), [1, 2]],
)
def test_init_custom_connection_grid_size(small_env, small_graph, connection_grid_size):
    env = InitialMapping(0.5, connection_grid_size=connection_grid_size)
    assert nx.is_isomorphic(env._connection_graph, small_graph)
    assert (env._state["connection_graph_matrix"] == np.array([[0, 1], [1, 0]])).all()


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder(), EpisodeRewarder(), SingleStepRewarder()]
)
def test_init_custom_rewarder(rewarder):
    env = InitialMapping(1, connection_grid_size=(2, 2), rewarder=rewarder)
    assert env.rewarder == rewarder


@pytest.mark.parametrize(
    "connection_graph_matrix,connection_grid_size",
    [("test", None), (None, "test"), ("test", "test")],
)
def test_parse_connection_graph_warnings(
    small_graph, connection_graph_matrix, connection_grid_size
):
    with pytest.warns(UserWarning):
        InitialMapping._parse_connection_graph(
            connection_graph=small_graph,
            connection_graph_matrix=connection_graph_matrix,
            connection_grid_size=connection_grid_size,
        )


def test_parse_connection_graph_warnings2():
    with pytest.warns(UserWarning):
        InitialMapping._parse_connection_graph(
            connection_graph=None,
            connection_graph_matrix=np.zeros([2, 2]),
            connection_grid_size="test",
        )


def test_parse_connection_graph_exception():
    msg = "No valid arguments for instantiation of the initial mapping environment were"
    msg += " provided."
    with pytest.raises(ValueError, match=msg):
        InitialMapping._parse_connection_graph(
            connection_graph=None,
            connection_graph_matrix=None,
            connection_grid_size=None,
        )
