import numpy as np
import pytest
from scipy.sparse import csr_matrix
from stable_baselines3.common.env_checker import check_env

from qgym.envs import InitialMapping
from qgym.envs.initial_mapping import BasicRewarder


def test_validity() -> None:
    env = InitialMapping(
        connection_grid_size=(3, 3), interaction_graph_edge_probability=0.5
    )
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True


"""
Tests for the basic rewarder
"""


def test_initialisation() -> None:
    rewarder = BasicRewarder()
    assert True


@pytest.mark.parametrize(
    "old_state,action,new_state,expected_reward",
    [
        # illegal action
        (
            {"logical_qubits_mapped": {2}, "physical_qubits_mapped": {1}},
            np.array([1, 2]),
            {},
            -10,
        ),
        # both graphs no edges, first step
        (
            {"logical_qubits_mapped": {}, "physical_qubits_mapped": {}},
            np.array([1, 1]),
            {
                "connection_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "mapping": np.array([1, 0]),
                "logical_qubits_mapped": {1},
                "physical_qubits_mapped": {1},
            },
            0,
        ),
        # both graphs no edges, last step
        (
            {"logical_qubits_mapped": {1}, "physical_qubits_mapped": {1}},
            np.array([2, 2]),
            {
                "connection_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "mapping": np.array([1, 2]),
                "logical_qubits_mapped": {1, 2},
                "physical_qubits_mapped": {1, 2},
            },
            0,
        ),
        # both graphs fully connected, first step
        (
            {"logical_qubits_mapped": {}, "physical_qubits_mapped": {}},
            np.array([1, 1]),
            {
                "connection_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "mapping": np.array([1, 0]),
                "logical_qubits_mapped": {1},
                "physical_qubits_mapped": {1},
            },
            0,
        ),
        # both graphs fully connected, last step
        (
            {"logical_qubits_mapped": {1}, "physical_qubits_mapped": {1}},
            np.array([2, 2]),
            {
                "connection_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "mapping": np.array([1, 2]),
                "logical_qubits_mapped": {1, 2},
                "physical_qubits_mapped": {1, 2},
            },
            10,
        ),
        # connection graph fully connected, interaction graph no edges, first step
        (
            {"logical_qubits_mapped": {}, "physical_qubits_mapped": {}},
            np.array([1, 1]),
            {
                "connection_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "mapping": np.array([1, 0]),
                "logical_qubits_mapped": {
                    1,
                },
                "physical_qubits_mapped": {
                    1,
                },
            },
            0,
        ),
        # connection graph fully connected, interaction graph no edges, last step
        (
            {"logical_qubits_mapped": {1}, "physical_qubits_mapped": {1}},
            np.array([2, 2]),
            {
                "connection_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "mapping": np.array([1, 2]),
                "logical_qubits_mapped": {1, 2},
                "physical_qubits_mapped": {1, 2},
            },
            0,
        ),
        # connection graph no edges, interaction fully connected, first step
        (
            {"logical_qubits_mapped": {}, "physical_qubits_mapped": {}},
            np.array([1, 1]),
            {
                "connection_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "mapping": np.array([1, 0]),
                "logical_qubits_mapped": {1},
                "physical_qubits_mapped": {1},
            },
            0,
        ),
        # connection graph no edges, interaction fully connected, last step
        (
            {"logical_qubits_mapped": {1}, "physical_qubits_mapped": {1}},
            np.array([2, 2]),
            {
                "connection_graph_matrix": csr_matrix([[0, 0], [0, 0]]),
                "interaction_graph_matrix": csr_matrix([[0, 1], [1, 0]]),
                "mapping": np.array([1, 2]),
                "logical_qubits_mapped": {1, 2},
                "physical_qubits_mapped": {1, 2},
            },
            -2,
        ),
    ],
)
def test_rewarder(old_state, action, new_state, expected_reward) -> None:
    rewarder = BasicRewarder()

    reward = rewarder.compute_reward(
        old_state=old_state, action=action, new_state=new_state
    )

    assert reward == expected_reward
