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


def test_validity() -> None:
    env = InitialMapping(
        connection_grid_size=(3, 3), interaction_graph_edge_probability=0.5
    )
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True


small_graph = nx.Graph()
small_graph.add_edge(0, 1)


@pytest.mark.parametrize(
    "mode,error_type,error_msg",
    [
        (1, TypeError, "'mode' must be a string, but was of type <class 'int'>"),
        ("test", ValueError, "The given render mode is not supported"),
    ],
)
def test_unsupported_render_mode(mode, error_type, error_msg):
    env = InitialMapping(0.5, connection_graph=small_graph)
    with pytest.raises(error_type, match=error_msg):
        env.render(mode=mode)
    env.close()


def test_init_custom_connection_graph():
    env = InitialMapping(0.5, connection_graph=small_graph)
    assert nx.is_isomorphic(env._connection_graph, small_graph)
    assert (env._state["connection_graph_matrix"] == np.array([[0, 1], [1, 0]])).all()


@pytest.mark.parametrize(
    "connection_graph_matrix",
    [np.array([[0, 1], [1, 0]]), [[0, 1], [1, 0]], csr_matrix([[0, 1], [1, 0]])],
)
def test_init_custom_connection_graph_matrix(connection_graph_matrix):
    env = InitialMapping(0.5, connection_graph_matrix=connection_graph_matrix)
    assert nx.is_isomorphic(env._connection_graph, small_graph)
    assert (env._state["connection_graph_matrix"] == np.array([[0, 1], [1, 0]])).all()


@pytest.mark.parametrize(
    "connection_grid_size",
    [(2, 1), [1, 2]],
)
def test_init_custom_connection_grid_size(connection_grid_size):
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
    "interaction_graph_edge_probability,connection_graph,connection_graph_matrix,connection_grid_size",
    [
        (0.5, small_graph, "test", None),
        (0.5, small_graph, None, "test"),
        (0.5, small_graph, "test", "test"),
        (0.5, None, np.zeros([2, 2]), "test"),
    ],
)
def test_init_warnings(
    interaction_graph_edge_probability,
    connection_graph,
    connection_graph_matrix,
    connection_grid_size,
):
    with pytest.warns(UserWarning):
        InitialMapping(
            interaction_graph_edge_probability=interaction_graph_edge_probability,
            connection_graph=connection_graph,
            connection_graph_matrix=connection_graph_matrix,
            connection_grid_size=connection_grid_size,
        )


graph_with_selfloop = nx.Graph()
graph_with_selfloop.add_edge(1, 1)


@pytest.mark.parametrize(
    "interaction_graph_edge_probability,connection_graph,connection_graph_matrix,connection_grid_size,rewarder,error_type,error_msg",
    [
        (
            "test",
            None,
            None,
            (2, 2),
            None,
            TypeError,
            "'interaction_graph_edge_probability' should be a real number, but was of type <class 'str'>",
        ),
        (
            -1,
            None,
            None,
            (2, 2),
            None,
            ValueError,
            "'interaction_graph_edge_probability' has an inclusive lower bound of 0, but was -1",
        ),
        (
            2,
            None,
            None,
            (2, 2),
            None,
            ValueError,
            "'interaction_graph_edge_probability' has an inclusive upper bound of 1, but was 2",
        ),
        (
            0.5,
            None,
            None,
            None,
            None,
            ValueError,
            "No valid arguments for instantiation of the initial mapping environment were provided.",
        ),
        (
            0.5,
            None,
            None,
            (2, 2),
            "test",
            TypeError,
            "'rewarder' must be an instance of Rewarder, but was of type <class 'str'>",
        ),
        (
            0.5,
            nx.Graph(),
            None,
            None,
            None,
            ValueError,
            "'connection_graph' has no nodes",
        ),
        (
            0.5,
            "test",
            None,
            None,
            None,
            TypeError,
            "'connection_graph' is not an instance of networkx.Graph, but was of type <class 'str'>",
        ),
        (
            0.5,
            graph_with_selfloop,
            None,
            None,
            None,
            ValueError,
            "'connection_graph' contains selfloops",
        ),
    ],
)
def test_init_exceptions(
    interaction_graph_edge_probability,
    connection_graph,
    connection_graph_matrix,
    connection_grid_size,
    rewarder,
    error_type,
    error_msg,
):
    with pytest.raises(error_type, match=error_msg):
        InitialMapping(
            interaction_graph_edge_probability=interaction_graph_edge_probability,
            connection_graph=connection_graph,
            connection_graph_matrix=connection_graph_matrix,
            connection_grid_size=connection_grid_size,
            rewarder=rewarder,
        )
