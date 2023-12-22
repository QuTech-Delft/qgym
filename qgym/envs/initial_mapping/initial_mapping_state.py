"""This module contains the :class:`~qgym.envs.initial_mapping.InitialMappingState`
class.

This :class:`~qgym.envs.InitialMapping` represents the :class:`~qgym.templates.State` of
the :class:`~qgym.envs.InitialMapping` environment.

Usage:
    >>> from qgym.envs.initial_mapping.initial_mapping_state import InitialMappingState
    >>> import networkx as nx
    >>> connection_graph = nx.grid_graph((3,3))
    >>> state = InitialMappingState(connection_graph, 0.5)

"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Union, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from qgym import spaces
from qgym.templates.state import State


class InitialMappingState(
    State[Dict[str, Union[NDArray[np.int_], NDArray[np.float_]]], NDArray[np.int_]]
):
    """The :class:`~qgym.envs.initial_mapping.InitialMappingState` class."""

    __slots__ = (
        "steps_done",
        "graphs",
        "mapping",
        "mapping_dict",
        "mapped_qubits",
    )

    def __init__(
        self, connection_graph: nx.Graph, interaction_graph_edge_probability: float
    ) -> None:
        """Init of the :class:`~qgym.envs.initial_mapping.InitialMappingState` class.

        Args:
            connection_graph: `networkx Graph <https://networkx.org/documentation/stable/reference/classes/graph.html>`_
                representation of the QPU topology. Each node represents a physical
                qubit and each edge represents a connection in the QPU topology.
            interaction_graph_edge_probability: Probability that an edge between any
                pair of qubits in the random interaction graph exists. The interaction
                graph will have the same amount of nodes as the connection graph. Nodes
                without any interactions can be seen as 'null' nodes. Must be a value in
                the range $[0,1]$.
        """
        # Create a random connection graph with `n_nodes` and with edges existing with
        # probability `interaction_graph_edge_probability` (nodes without connections
        # can be seen as 'null' nodes)
        interaction_graph = nx.fast_gnp_random_graph(
            connection_graph.number_of_nodes(),
            interaction_graph_edge_probability,
        )

        self.steps_done: int = 0
        """Number of steps done since the last reset."""

        self.fidelity = False  # whether edges include fidelity
        for _, _, weight in connection_graph.edges.data("weight"):
            if not isinstance(weight, int):
                self.fidelity = True
                break
        if self.fidelity:
            connection = {
                "graph": deepcopy(connection_graph),
                "matrix": nx.to_numpy_array(connection_graph, dtype=np.float_),
            }
        else:
            connection = {
                "graph": deepcopy(connection_graph),
                "matrix": nx.to_numpy_array(connection_graph, dtype=np.int8),
            }
        self.graphs = {
            "connection": connection,
            "interaction": {
                "graph": deepcopy(interaction_graph),
                "matrix": nx.to_numpy_array(interaction_graph, dtype=np.int8).flatten(),
                "edge_probability": interaction_graph_edge_probability,
            },
        }
        """Dictionary containing the graph and matrix representations of the both the
        interaction graph and connection graph.
        """
        self.mapping = np.full(self.n_nodes, self.n_nodes, dtype=np.int_)
        """Array of which the index represents a physical qubit, and the value a virtual
        qubit. A value of ``n_nodes + 1`` represents the case when nothing is mapped to
        the physical qubit yet.
        """
        self.mapping_dict: dict[int, int] = {}
        """Dictionary that maps logical qubits (keys) to physical qubits (values)."""
        self.mapped_qubits: dict[str, set[int]] = {"physical": set(), "logical": set()}
        """Dictionary with two sets containing mapped physical and logical qubits."""

    def create_observation_space(self) -> spaces.Dict:
        """Create the corresponding observation space.

        Returns:
            Observation space in the form of a :class:`~qgym.spaces.Dict` space
            containing:

            * :class:`~qgym.spaces.MultiDiscrete` space representing the mapping.
            * :class:`~qgym.spaces.MultiBinary` representing the interaction matrix.
        """
        mapping_space = spaces.MultiDiscrete(
            nvec=[self.n_nodes + 1] * self.n_nodes, rng=self.rng
        )

        if self.fidelity:
            interaction_matrix_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.n_nodes * self.n_nodes,),
                dtype=np.float_,
                rng=self.rng,
            )
            return spaces.Dict(
            rng=self.rng,
            mapping=mapping_space,
            interaction_matrix=interaction_matrix_space,
        )
        else:
            return spaces.Dict(
            rng=self.rng,
            mapping=mapping_space,
            interaction_matrix=spaces.MultiBinary(
                self.n_nodes**2, rng=self.rng
            ),
        )
            
    def reset(
        self,
        *,
        seed: int | None = None,
        interaction_graph: nx.Graph | None = None,
        **_kwargs: Any,
    ) -> InitialMappingState:
        """Reset the state and set a new interaction graph.

        To be used after an episode is finished.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call i.e., before any learning is done.
            interaction_graph: Interaction graph to be used for the next iteration, if
            ``None`` a random interaction graph will be created.
            _kwargs: Additional options to configure the reset.

        Returns:
            (self) New initial state.
        """
        if seed is not None:
            self.seed(seed)

        # Reset the state
        if interaction_graph is None:
            self.graphs["interaction"]["graph"] = nx.fast_gnp_random_graph(
                self.n_nodes, self.graphs["interaction"]["edge_probability"]
            )
        else:
            self.graphs["interaction"]["graph"] = deepcopy(interaction_graph)

        self.graphs["interaction"]["matrix"] = nx.to_numpy_array(
            self.graphs["interaction"]["graph"], dtype=np.int8
        ).flatten()

        self.steps_done = 0
        self.mapping = np.full(self.n_nodes, self.n_nodes)
        self.mapping_dict = {}
        self.mapped_qubits = {"physical": set(), "logical": set()}

        return self

    def update_state(self, action: NDArray[np.int_]) -> InitialMappingState:
        """Update the state (in place) of this environment using the given action.

        Args:
            action: Mapping action to be executed.

        Returns:
            Self.
        """
        # Increase the step number
        self.steps_done += 1

        # update state based on the given action
        physical_qubit, logical_qubit = action

        if (
            physical_qubit in self.mapped_qubits["physical"]
            or logical_qubit in self.mapped_qubits["logical"]
        ):
            return self

        self.mapping[physical_qubit] = logical_qubit
        self.mapping_dict[logical_qubit] = physical_qubit
        self.mapped_qubits["physical"].add(physical_qubit)
        self.mapped_qubits["logical"].add(logical_qubit)
        return self

    def obtain_observation(self) -> dict[str, NDArray[np.int_]]:
        """Obtain an observation based on the current state.

        Returns:
            Observation based on the current state.
        """
        return {
            "mapping": self.mapping,
            "interaction_matrix": self.graphs["interaction"]["matrix"],
        }

    def is_done(self) -> bool:
        """Determine if the state is done or not.

        Returns:
            Boolean value stating whether we are in a final state.
        """
        return bool(len(self.mapping_dict) == self.n_nodes)

    def obtain_info(self) -> dict[str, Any]:
        """Obtain additional information.

        Returns:
            Optional debugging info for the current state.
        """
        return {
            "Steps done": self.steps_done,
            "Mapping": self.mapping,
            "Mapping Dictionary": self.mapping_dict,
            "Mapped Qubits": self.mapped_qubits,
        }

    @property
    def n_nodes(self) -> int:
        """The number of physical qubits."""
        return cast(int, self.graphs["connection"]["graph"].number_of_nodes())
