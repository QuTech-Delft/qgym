from copy import deepcopy
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from qgym.envs import InitialMapping
from qgym.envs.initial_mapping_rewarders import (
    BasicRewarder,
    EpisodeRewarder,
    SingleStepRewarder,
)
from qgym.rewarder import Rewarder
from scipy.sparse import csr_matrix
from stable_baselines3.common.env_checker import check_env


def test_validity() -> None:
    env = InitialMapping(
        connection_grid_size=(3, 3), interaction_graph_edge_probability=0.5
    )
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True


def _episode_generator(
    adjacency_matrices: Dict[str, csr_matrix]
) -> Iterator[Tuple[Dict[str, Any], NDArray[np.int_], Dict[str, Any]]]:

    old_state = deepcopy(adjacency_matrices)

    old_state["mapping_dict"] = {}
    old_state["logical_qubits_mapped"] = set()
    old_state["physical_qubits_mapped"] = set()

    new_state = deepcopy(old_state)

    action = np.array([0, 0])
    _perform_action(new_state, action)

    yield (old_state, action, new_state)

    for i in range(1, adjacency_matrices["connection_graph_matrix"].shape[0]):
        _perform_action(old_state, action)
        action = np.array([i, i])
        _perform_action(new_state, action)

        yield (old_state, action, new_state)


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
    assert reward == -10


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
            [0, 10, 30],
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
            [0, -2, -6],
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
            [0, 10, 20],
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
            [0, -2, -4],
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
            [0, 0, 30],
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
            [0, 0, -6],
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
