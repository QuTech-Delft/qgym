"""This module contains the ``InitialMappingState`` class.

This ``InitialMappingState`` represents the ``State`` of the ``InitialMapping``
environment.

Usage:
    >>> from qgym.envs.initial_mapping.initial_mapping_state import InitialMappingState
    >>> import networkx as nx
    >>> connection_graph = nx.grid_graph((3,3))
    >>> state = RoutingState(
    >>>             max_interaction_gates = 100,
    >>>             max_observation_reach = 20,
    >>>             connection_graph = connection_graph,
    >>>             observation_booleans_flag = True,
    >>>             observation_connection_flag = False,
    >>>             )

"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Set

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.templates.state import State


class InitialMappingState(State[Dict[str, NDArray[np.int_]], NDArray[np.int_]]):
    """The ``InitialMappingState`` class.

    :ivar steps_done: Number of steps done since the last reset.
    :ivar num_nodes: Number of nodes in the connection graph. Represent the of physical
        qubits.
    :ivar graphs: Dictionary containing the graph and matrix representations of the
        both the interaction graph and connection graph.
    :ivar mapping: Array of which the index represents a physical qubit, and the value a
        virtual qubit. A value of ``num_nodes + 1`` represents the case when nothing is
        mapped to the physical qubit yet.
    :ivar mapping_dict: Dictionary that maps logical qubits (keys) to physical qubits
        (values).
    :ivar mapped_qubits: Dictionary with a two ``Set``s containing all mapped physical
        and logical qubits.
    """

    def __init__(
        self, connection_graph: nx.Graph, interaction_graph_edge_probability: float
    ) -> None:
        """Init of the ``InitialMappingState`` class.

        :param connection_graph: ``networkx`` graph representation of the QPU topology.
            Each node represents a physical qubit and each edge represents a connection
            in the QPU topology.
        :param interaction_graph_edge_probability: Probability that an edge between any
            pair of qubits in the random interaction graph exists. The interaction
            graph will have the same amount of nodes as the connection graph. Nodes
            without any interactions can be seen as 'null' nodes. Must be a value in the
            range $[0,1]$.
        """
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
        self.mapping_dict: Dict[int, int] = {}
        self.mapped_qubits: Dict[str, Set[int]] = {"physical": set(), "logical": set()}

    def create_observation_space(self) -> qgym.spaces.Dict:
        """Create the corresponding observation space.

        :returns: Observation space in the form of a ``qgym.spaces.Dict`` space
            containing:

            * ``qgym.spaces.MultiDiscrete`` space representing the mapping.
            * ``qgym.spaces.Box`` representing the interaction matrix.
        """
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
        """Reset the state and set a new interaction graph.

        To be used after an episode sis finished.

        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call i.e., before any learning is done.
        :param interaction_graph: Interaction graph to be used for the next iteration,
            if ``None`` a random interaction graph will be created.
        :param _kwargs: Additional options to configure the reset.
        :return: (self) New initial state.
        """
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
        for node1, node2 in self.graphs["connection"]["graph"].edges():
            weight = self.rng.gamma(2, 2) / 4
            self.graphs["connection"]["graph"].edges[node1, node2]["weight"] = weight
        self.graphs["connection"]["matrix"] = nx.to_scipy_sparse_array(
            self.graphs["connection"]["graph"]
        )

        for node1, node2 in self.graphs["interaction"]["graph"].edges():
            weight = self.rng.gamma(2, 2) / 4
            self.graphs["interaction"]["graph"].edges[node1, node2]["weight"] = weight
        self.graphs["interaction"]["matrix"] = nx.to_scipy_sparse_array(
            self.graphs["interaction"]["graph"]
        )

    def update_state(self, action: NDArray[np.int_]) -> InitialMappingState:
        """Update the state of this environment using the given action.

        :param action: Mapping action to be executed.
        """
        # Increase the step number
        self.steps_done += 1

        # update state based on the given action
        physical_qubit_index = action[0]
        logical_qubit_index = action[1]

        if (
            physical_qubit_index in self.mapped_qubits["physical"]
            or logical_qubit_index in self.mapped_qubits["logical"]
        ):
            return self

        self.mapping[physical_qubit_index] = logical_qubit_index
        self.mapping_dict[logical_qubit_index] = physical_qubit_index
        self.mapped_qubits["physical"].add(physical_qubit_index)
        self.mapped_qubits["logical"].add(logical_qubit_index)
        return self

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
