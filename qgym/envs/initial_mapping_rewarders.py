"""
Environment and rewarder for training an RL agent on the initial mapping problem of OpenQL.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from qgym import Rewarder

# Define some colors used during rendering
WHITE = (225, 225, 225)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
RED = (225, 0, 0)
GREEN = (0, 225, 0)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 0, 225)


class BasicRewarder(Rewarder):
    """
    Basic rewarder for the InitialMapping environment.
    """

    def __init__(
        self,
        illegal_action_penalty: Optional[float] = -10,
        reward_per_edge: Optional[float] = 5,
        penalty_per_edge: Optional[float] = -1,
    ) -> None:
        """
        Initialize the reward range and set the rewards and penaltyies

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param reward_per_edge: reward for performing a 'good' action
        :param new_state: penalty for performing a 'bad' action
        """
        self._reward_range = (-float("inf"), float("inf"))
        self._illegal_action_penalty = illegal_action_penalty
        self._reward_per_edge = reward_per_edge
        self._penalty_per_edge = penalty_per_edge

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ):
        """
        Compute a reward, based on the current state, and the connection and interaction graphs.

        :param old_state: State of the InitialMapping before the current action.
        :param action: Action that has just been taken
        :param new_state: Updated state of the InitialMapping
        """

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        mapped_edges = self._get_mapped_edges(new_state)

        reward = 0.0
        for i, j in mapped_edges:
            if new_state["connection_graph_matrix"][i, j] == 0:
                reward += self._penalty_per_edge
            else:
                reward += self._reward_per_edge

        return reward

    def _get_mapped_edges(self, new_state: Dict[Any, Any]) -> list[list[int, int]]:
        """
        Computes a list of mapped edges of the new state

        :param new_state: Updated state of the InitialMapping
        """

        mapping = self._get_mapping_dct(new_state["mapping"])

        mapped_edges = []

        for logical_qubit_idx, physical_qubit_idx in mapping.items():
            neighbours = self._get_neighbours(
                logical_qubit_idx, new_state["interaction_graph_matrix"]
            )

            mapped_neighbours = neighbours & new_state["logical_qubits_mapped"]

            for mapped_neighbour in mapped_neighbours:
                mapped_edges.append(
                    [physical_qubit_idx - 1, mapping[mapped_neighbour] - 1]
                )

        return mapped_edges

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: Dict[Any, Any]) -> bool:
        """
        Checks if the given action is illegal, i.e. checks if qubits are mapped multiple times.

        :param action: Action that has just been taken
        :param old_state: State of the InitialMapping before the current action.
        """
        return (
            action[0] in old_state["physical_qubits_mapped"]
            and action[1] in old_state["logical_qubits_mapped"]
        )

    @staticmethod
    def _get_mapping_dct(mapping_array: NDArray[np.int_]) -> Dict[int, int]:
        """
        Converts the mapping array of a state to a dictionary.
        For example np.ndarray([3, 0, 2, 0]) --> {2 : 3, 3 : 1}

        :param mapping_array: mapping_array of the state to be converted.
        """
        mapping_dct = {}
        for i, logical_qubit_index in enumerate(mapping_array):
            physical_qubit_index = i + 1
            if logical_qubit_index != 0:
                mapping_dct[logical_qubit_index] = physical_qubit_index
        return mapping_dct

    def _get_neighbours(self, qubit_idx: int, adjacency_matrix: csr_matrix) -> set:
        """
        Computes a set of neighbours of a given qubit (i.e. a set of nodes whith with this node has a connection.)

        :param qubit_idx: index of the qubit of which the neihbours are computed.
        :param adjacency_matrix: adjacency matrix of a graph
        """
        neighbours = set()
        for i in range(adjacency_matrix.shape[0]):
            neighbour_idx = i + 1
            if self._are_neighbours((qubit_idx, neighbour_idx), adjacency_matrix):
                neighbours.add(neighbour_idx)
        return neighbours

    @staticmethod
    def _are_neighbours(
        qubit_idxs: Tuple[int, int], adjacency_matrix: csr_matrix
    ) -> bool:
        """
        Checks if two qubits are neighbours

        :param qubit_idxs: indexes of the qubits to check
        :param adjacency_matrix: adjacency matrix of a graph
        """
        return adjacency_matrix[qubit_idxs[0] - 1, qubit_idxs[1] - 1] != 0



