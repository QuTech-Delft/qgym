import warnings
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Tuple

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from stable_baselines3.common.env_checker import check_env

from qgym.envs import InitialMapping
from qgym.envs.initial_mapping.initial_mapping_rewarders import (
    BasicRewarder,
    EpisodeRewarder,
    SingleStepRewarder,
)
from qgym.rewarder import Rewarder


def test_validity() -> None:
    env = InitialMapping(
        connection_grid_size=(3, 3), interaction_graph_edge_probability=0.5
    )
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True


def _episode_generator(
    adjacency_matrices: Dict[str, Any]
) -> Iterator[Tuple[Dict[str, Any], NDArray[np.int_], Dict[str, Any]]]:

    old_state = deepcopy(adjacency_matrices)

    old_state["mapping_dict"] = {}
    old_state["logical_qubits_mapped"] = set()
    old_state["physical_qubits_mapped"] = set()

    new_state = deepcopy(old_state)

    action = np.array([0, 0])
    _perform_action(new_state, action)

    yield old_state, action, new_state

    for i in range(1, adjacency_matrices["connection_graph_matrix"].shape[0]):
        _perform_action(old_state, action)
        action = np.array([i, i])
        _perform_action(new_state, action)

        yield old_state, action, new_state


def _perform_action(state: Dict[str, Any], action: NDArray[np.int_]) -> None:
    state["mapping_dict"][action[0]] = action[1]
    state["logical_qubits_mapped"].add(action[0])
    state["physical_qubits_mapped"].add(action[1])


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder(), SingleStepRewarder(), EpisodeRewarder()]
)
def test_illegal_actions(rewarder: Rewarder) -> None:

    old_state = {"logical_qubits_mapped": {2}, "physical_qubits_mapped": {1}}
    action = np.array([1, 2])
    new_state = {}

    reward = rewarder.compute_reward(
        old_state=old_state, action=action, new_state=new_state
    )
    assert reward == -100


"""
Tests for the basic rewarder
"""


@pytest.mark.parametrize(
    "adjacency_matrices,rewards",
    [  # Both graphs empty
        (
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
            },
            [0, 0, 0],
        ),
        # Both graphs fully connected
        (
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
            },
            [0, 5, 15],
        ),
        (
            # connection graph fully connected, interaction graph no edges
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
            },
            [0, 0, 0],
        ),
        (
            # connection graph no edges, interaction fully connected
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
            },
            [0, -1, -3],
        ),
    ],
)
def test_basic_rewarder(
    adjacency_matrices: Dict[str, csr_matrix], rewards: List[float]
) -> None:
    episode_generator = _episode_generator(adjacency_matrices)

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
    "adjacency_matrices,rewards",
    [  # Both graphs empty
        (
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
            },
            [0, 0, 0],
        ),
        # Both graphs fully connected
        (
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
            },
            [0, 5, 10],
        ),
        (
            # connection graph fully connected, interaction graph no edges
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
            },
            [0, 0, 0],
        ),
        (
            # connection graph no edges, interaction fully connected
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
            },
            [0, -1, -2],
        ),
    ],
)
def test_single_step_rewarder(
    adjacency_matrices: Dict[str, csr_matrix], rewards: List[float]
) -> None:
    episode_generator = _episode_generator(adjacency_matrices)

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
    "adjacency_matrices,rewards",
    [  # Both graphs empty
        (
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
            },
            [0, 0, 0],
        ),
        # Both graphs fully connected
        (
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
            },
            [0, 0, 15],
        ),
        (
            # connection graph fully connected, interaction graph no edges
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
            },
            [0, 0, 0],
        ),
        (
            # connection graph no edges, interaction fully connected
            {
                "connection_graph_matrix": csr_matrix(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                ),
                "interaction_graph_matrix": csr_matrix(
                    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
                ),
            },
            [0, 0, -3],
        ),
    ],
)
def test_episode_step_rewarder(
    adjacency_matrices: Dict[str, csr_matrix], rewards: List[float]
) -> None:
    episode_generator = _episode_generator(adjacency_matrices)

    rewarder = EpisodeRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]


small_graph = nx.Graph()
small_graph.add_edge(0, 1)


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
            "interaction_graph_edge_probability should be a real number, but was of type <class 'str'>.",
        ),
        (
            -1,
            None,
            None,
            (2, 2),
            None,
            ValueError,
            "interaction_graph_edge_probability has an inclusive lower bound of 0, but was -1.",
        ),
        (
            2,
            None,
            None,
            (2, 2),
            None,
            ValueError,
            "interaction_graph_edge_probability has an inclusive upper bound of 1, but was 2.",
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
            "The given rewarder was not an instance of Rewarder.",
        ),
        (
            0.5,
            nx.Graph(),
            None,
            None,
            None,
            ValueError,
            "The given 'connection_graph' has no nodes.",
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
