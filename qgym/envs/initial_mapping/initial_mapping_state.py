from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.state import State


class InitialMappingState(State[Dict[str, NDArray[np.int_]], NDArray[np.int_]]):
    def __init__(
        self, connection_graph: nx.Graph, interaction_graph_edge_probability: float
    ) -> None:
        # Create a random connection graph with `num_nodes` and with edges existing with
        # probability `interaction_graph_edge_probability` (nodes without connections
        # can be seen as 'null' nodes)
        interaction_graph = nx.fast_gnp_random_graph(
            connection_graph.number_of_nodes(),
            interaction_graph_edge_probability,
        )

        self.steps_done = 0
        self.num_nodes = connection_graph.number_of_nodes()
        self.graphs = {
            "connection": {
                "graph": deepcopy(connection_graph),
                "matrix": nx.to_scipy_sparse_array(connection_graph),
            },
            "interaction": {
                "graph": deepcopy(interaction_graph),
                "matrix": nx.to_scipy_sparse_array(interaction_graph),
                "edge_probability": interaction_graph_edge_probability,
            },
        }
        self.mapping = np.full(self.num_nodes, self.num_nodes)
        self.mapping_dict = {}
        self.mapped_qubits = {"physical": set(), "logical": set()}

    def create_observation_space(self) -> qgym.spaces.Dict:
        mapping_space = qgym.spaces.MultiDiscrete(
            nvec=[self.num_nodes + 1] * self.num_nodes, rng=self.rng
        )
        interaction_matrix_space = qgym.spaces.Box(
            low=0,
            high=np.iinfo(np.int64).max,
            shape=(self.num_nodes * self.num_nodes,),
            dtype=np.int64,
        )
        observation_space = qgym.spaces.Dict(
            rng=self.rng,
            mapping=mapping_space,
            interaction_matrix=interaction_matrix_space,
        )
        return observation_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        interaction_graph: Optional[nx.Graph] = None,
        **_kwargs: Any,
    ) -> InitialMappingState:

        if seed is not None:
            self.seed(seed)

        # Reset the state
        if interaction_graph is None:
            self.graphs["interaction"]["graph"] = nx.fast_gnp_random_graph(
                self.num_nodes, self.graphs["interaction"]["edge_probability"]
            )
        else:
            self.graphs["interaction"]["graph"] = deepcopy(interaction_graph)

        self.graphs["interaction"]["matrix"] = nx.to_scipy_sparse_array(
            self.graphs["interaction"]["graph"]
        ).toarray()

        self.steps_done = 0
        self.mapping = np.full(self.num_nodes, self.num_nodes)
        self.mapping_dict = {}
        self.mapped_qubits = {"physical": set(), "logical": set()}

        return self

    def add_random_edge_weights(self) -> None:
        """Add random weights to the connection graph and interaction graph."""
        for (node1, node2) in self.graphs["connection"]["graph"].edges():
            weight = self.rng.gamma(2, 2) / 4
            self.graphs["connection"]["graph"].edges[node1, node2]["weight"] = weight
        self.graphs["connection"]["matrix"] = nx.to_scipy_sparse_array(
            self.graphs["connection"]["graph"]
        )

        for (node1, node2) in self.graphs["interaction"]["graph"].edges():
            weight = self.rng.gamma(2, 2) / 4
            self.graphs["interaction"]["graph"].edges[node1, node2]["weight"] = weight
        self.graphs["interaction"]["matrix"] = nx.to_scipy_sparse_array(
            self.graphs["interaction"]["graph"]
        )

    def update_state(self, action: NDArray[np.int_]) -> None:
        """Update the state of this environment using the given action.

        :param action: Mapping action to be executed.
        """
        # Increase the step number
        self.steps_done += 1

        # update state based on the given action
        physical_qubit_index = action[0]
        logical_qubit_index = action[1]
        if (
            physical_qubit_index not in self.mapped_qubits["physical"]
            and logical_qubit_index not in self.mapped_qubits["logical"]
        ):
            self.mapping[physical_qubit_index] = logical_qubit_index
            self.mapping_dict[logical_qubit_index] = physical_qubit_index
            self.mapped_qubits["physical"].add(physical_qubit_index)
            self.mapped_qubits["logical"].add(logical_qubit_index)

    def obtain_observation(self) -> Dict[str, NDArray[np.int_]]:
        """:return: Observation based on the current state."""
        return {
            "mapping": self.mapping,
            "interaction_matrix": self.graphs["interaction"]["matrix"].flatten(),
        }

    def is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        return bool(len(self.mapped_qubits["physical"]) == self.num_nodes)

    def obtain_info(self) -> Dict[str, Any]:
        """:return: Optional debugging info for the current state."""
        return {"Steps done": self.steps_done}
