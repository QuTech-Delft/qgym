# r"""This module contains an environment for training an RL agent on the routing
# problem of OpenQL. The routing problem is aimed at enabling to execute the quantum circuit
# by putting those physical qubits into connection that have an interaction in the quantum
# circuit. This problem arises when there are mismatches between the interaction graph and the
# QPU-topology in the initial mapping.
# The quantum circuit is represented as an **interaction graph**, where each node represent a
# qubit and each edge represent an interaction between two qubits as defined by the circuit
# (See the example below). The QPU structure is called the **connection graph**. In the connection
# graph each node represents a physical qubit and each edge represent a connection between
# two qubits in the QPU.


# .. code-block:: console

#               QUANTUM CIRCUIT                        INTERACTION GRAPH
#            ┌───┐               ┌───┐
#     |q3>───┤ R ├───┬───────────┤ M ╞══                 q1 ────── q2
#            └───┘   │           └───┘                            ╱
#            ┌───┐ ┌─┴─┐         ┌───┐                           ╱
#     |q2>───┤ R ├─┤ X ├───┬─────┤ M ╞══                        ╱
#            └───┘ └───┘   │     └───┘                         ╱
#            ┌───┐       ┌─┴─┐   ┌───┐                        ╱
#     |q1>───┤ R ├───┬───┤ X ├───┤ M ╞══                     ╱
#            └───┘   │   └───┘   └───┘                      ╱
#            ┌───┐ ┌─┴─┐         ┌───┐                     ╱
#     |q0>───┤ R ├─┤ X ├─────────┤ M ╞══                q3 ─────── q4
#            └───┘ └───┘         └───┘


# A SWAP-gate changes the mapping from logical qubits to physical qubits at a certain point in
# the circuit, and thereby allows to solve mismatchings from the initial mapping.
# The goal is to place SWAP-gates in the quantum circuit to fixed the mismatches. The least amount
# of SWAP-gates is preferred. In more advanced setups, also different factors can be taken into
# account, like the fidelity of edge in the QPU etc.


# State Space:
#     The state space is described by a ``RoutingState`` with the following
#     attributes:

#     * `steps_done`: Number of steps done since the last reset.
#     * `num_nodes`: Number of *physical* qubits.
#     * `graphs`: Dictionary containing the graph and matrix representation of the connection graph.
#     * `mapping`: Array of which the index represents a physical qubit, and the value a
#       virtual qubit. A value of ``num_nodes + 1`` represents the case when nothing is
#       mapped to the physical qubit yet.
#     * `mapping_dict`: Dictionary that maps logical qubits (keys) to physical qubit
#       (values).
#     * `mapped_qubits`: Dictionary with a two Sets containing all mapped physical and
#       logical qubits.

# Observation Space:
#     The observation space is a ``qgym.spaces.Dict`` with 2 entries:

#     * `m_execution_array`: Array with Boolean values for the upcoming m multiple qubit gates in the quantum circuit.
#       A Boolean value True in the array indicates that, given the current mapping and the connection graph, that
#       multiple qubit gate can be executed.
#     * `mapping`: The current state of the mapping.

# Action Space:
#     A valid action is a tuple of integers  $(i,j,k)$, such that $i$ indicates where in
#     the circuit (i.e. after the i-th gate, counting from zero) to put in a SWAP-gate on
#     physical qubits $j$ and $k$.
#     An action is legal when:
#     #. physical qubit $j$ does not equal physical qubit $k$.
#     #. physical qubits $j$ and $k$ have a connection between them in the connection graph.

# TODO: create Examples


# """
# import warnings
# from copy import deepcopy
# from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

# import networkx as nx
# import numpy as np
# from networkx import Graph, grid_graph
# from numpy.typing import ArrayLike, NDArray

# import qgym.spaces
# from qgym.envs.routing.routing_rewarders import (
#     BasicRewarder,  # TODO: define BasicRewarder
# )
# from qgym.envs.routing.routing_state import RoutingState  # TODO: define RoutingState

# # TODO: Do we need a visualiser for routing?
# # from qgym.envs.initial_mapping.initial_mapping_visualiser import (
# #    InitialMappingVisualiser,
# # )
# from qgym.templates import Environment, Rewarder
# from qgym.utils.input_validation import (  # TODO: What utils do we need for routing?
#     check_adjacency_matrix,
#     check_graph_is_valid_topology,
#     check_instance,
#     check_real,
# )

# Gridspecs = Union[List[Union[int, Iterable[int]]], Tuple[Union[int, Iterable[int]]]]


# class Routing(Environment[Dict[str, NDArray[np.int_]], NDArray[np.int_]]):
#     """RL environment for the routing problem of OpenQL."""

#     def __init__(
#         self,
#         machine_properties: Union[Mapping[str, Any], str, MachineProperties],
#         # TODO: what do we need instead of the interaction graph?
#         *,
#         connection_graph: Optional[Graph] = None,
#         connection_graph_matrix: Optional[ArrayLike] = None,
#         connection_grid_size: Optional[Gridspecs] = None,
#         rewarder: Optional[Rewarder] = None,
#     ) -> None:
#         """Initialize the action space, observation space, and initial states.
#         #TODO: Write appropriate doc-string for Routing-environment!

#         Furthermore, the connection graph

#         The supported render modes of this environment are "human" and "rgb_array". #TODO: what do these render-modes entail?

#         :param connection_graph: ``networkx`` graph representation of the QPU topology.
#             Each node represents a physical qubit and each node represents a connection
#             in the QPU topology.
#         :param connection_graph_matrix: Adjacency matrix representation of the QPU
#             topology.
#         :param connection_grid_size: Size of the connection graph when the connection
#             graph has a grid topology. For more information on the allowed values and
#             types, see ``networkx`` `grid_graph`_ documentation.
#         :param rewarder: Rewarder to use for the environment. Must inherit from
#             ``qgym.Rewarder``. If ``None`` (default), then ``BasicRewarder`` is used.

#         .. _grid_graph: https://networkx.org/documentation/stable/reference/generated/
#             networkx.generators.lattice.grid_graph.html#grid-graph
#         """
#         connection_graph = self._parse_connection_graph(
#             connection_graph=connection_graph,
#             connection_graph_matrix=connection_graph_matrix,
#             connection_grid_size=connection_grid_size,
#         )

#         self._rewarder = self._parse_rewarder(rewarder)

#         # Define internal attributes
#         self._state = RoutingState(
#             connection_graph,  # TODO: what else do we need to define the internal attributes?
#         )
#         self.observation_space = (
#             self._state.create_observation_space()
#         )  # TODO: how does creat_observation_space() work.
#         # Define attributes defined in parent class
#         # TODO: how does the action_space work? What to do we the below command?
#         self.action_space = qgym.spaces.MultiDiscrete(
#             nvec=[self._state.num_nodes, self._state.num_nodes], rng=self.rng
#         )

#         # TODO: what is the role of this metadata?
#         self.metadata = {"render.modes": ["human", "rgb_array"]}

#         # TODO: self._visualiser = RoutingVisualiser(connection_graph)

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         return_info: bool = False,
#         # TODO: what should be returned for usage in a next iteration?
#         **_kwargs: Any,
#     ) -> Union[
#         Dict[str, NDArray[np.int_]],
#         Tuple[Dict[str, NDArray[np.int_]], Dict[str, Any]],
#     ]:
#         """Reset the state and set a new interaction graph.

#         To be used after an episode is finished.

#         :param seed: Seed for the random number generator, should only be provided
#             (optionally) on the first reset call i.e., before any learning is done.
#         :param return_info: Whether to receive debugging info. Default is ``False``.
#         #TODO: doc-string of what should be returned for usage in a next iteration?
#         :param _kwargs: Additional options to configure the reset.
#         :return: Initial observation and optionally debugging info.
#         """
#         # call super method for dealing with the general stuff
#         return super().reset(
#             seed=seed, return_info=return_info, interaction_graph=interaction_graph
#         )

#     def add_random_edge_weights(self) -> None:
#         """Add random weights to the connection graph and interaction graph."""
#         cast(RoutingState, self._state)

#     @staticmethod
#     def _parse_connection_graph(
#         *,
#         connection_graph: Any,
#         connection_graph_matrix: Any,
#         connection_grid_size: Any,
#     ) -> Graph:
#         """Parse the user input (given in ``__init__``) to create a connection graph.

#         :param connection_graph: ``networkx.Graph`` representation of the QPU topology.
#         :param connection_graph_matrix: Adjacency matrix representation of the QPU
#             topology
#         :param connection_grid_size: Size of the connection graph when the topology is a
#             grid.
#         :raise ValueError: When `connection_graph`, `connection_graph_matrix` and
#             `connection_grid_size` are all None.
#         :return: Connection graph as a ``networkx.Graph``.
#         """
#         if connection_graph is not None:
#             if connection_graph_matrix is not None:
#                 msg = "Both 'connection_graph' and 'connection_graph_matrix' were "
#                 msg += "given. Using 'connection_graph'."
#                 warnings.warn(msg)
#             if connection_grid_size is not None:
#                 msg = "Both 'connection_graph' and 'connection_grid_size' were given. "
#                 msg += "Using 'connection_graph'."
#                 warnings.warn(msg)

#             check_graph_is_valid_topology(connection_graph, "connection_graph")

#             # deepcopy the graphs for safety
#             return deepcopy(connection_graph)

#         if connection_graph_matrix is not None:
#             if connection_grid_size is not None:
#                 msg = "Both 'connection_graph_matrix' and 'connection_grid_size' were "
#                 msg += "given. Using 'connection_graph_matrix'."
#                 warnings.warn(msg)
#             return InitialMapping._parse_adjacency_matrix(connection_graph_matrix)
#         if connection_grid_size is not None:
#             # Generate connection grid graph
#             return grid_graph(connection_grid_size)

#         msg = "No valid arguments for instantiation of the initial mapping environment "
#         msg += "were provided."
#         raise ValueError(msg)

#     @staticmethod
#     def _parse_rewarder(rewarder: Union[Rewarder, None]) -> Rewarder:
#         """Parse the `rewarder` given by the user.

#         :param rewarder: ``Rewarder`` to use for the environment. If ``None``, then the
#             ``BasicRewarder`` with default settings is used.
#         :return: Rewarder.
#         """
#         if rewarder is None:
#             return BasicRewarder()
#         check_instance(rewarder, "rewarder", Rewarder)
#         return deepcopy(rewarder)

#     @staticmethod
#     def _parse_adjacency_matrix(connection_graph_matrix: ArrayLike) -> Graph:
#         """Parse a given connection graph adjacency matrix to its respective graph.

#         :param connection_graph_matrix: adjacency matrix representation of the QPU
#             topology.
#         :raise TypeError: When the provided matrix is not a valid adjacency matrix.
#         :return: Graph representation of the adjacency matrix.
#         """
#         connection_graph_matrix = check_adjacency_matrix(connection_graph_matrix)

#         return nx.from_numpy_array(connection_graph_matrix)
