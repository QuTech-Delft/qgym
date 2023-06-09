r"""This module contains an environment for training an RL agent on the routing problem
of OpenQL. The routing problem is aimed at enabling to execute the quantum circuit
by putting those physical qubits into connection that have an interaction in the quantum
circuit. This problem arises when there are mismatches between the interaction graph and
the QPU-topology in the initial mapping. The quantum circuit is represented as an
**interaction graph**, where each node represent a qubit and each edge represent an
interaction between two qubits as defined by the circuit (See the example below). The
QPU structure is called the **connection graph**. In the connection graph each node
represents a physical qubit and each edge represent a connection between two qubits in
the QPU.


.. code-block:: console

              QUANTUM CIRCUIT                        INTERACTION GRAPH
           ┌───┐               ┌───┐
    |q3>───┤ R ├───┬───────────┤ M ╞══                 q1 ────── q2
           └───┘   │           └───┘                            ╱
           ┌───┐ ┌─┴─┐         ┌───┐                           ╱
    |q2>───┤ R ├─┤ X ├───┬─────┤ M ╞══                        ╱
           └───┘ └───┘   │     └───┘                         ╱
           ┌───┐       ┌─┴─┐   ┌───┐                        ╱
    |q1>───┤ R ├───┬───┤ X ├───┤ M ╞══                     ╱
           └───┘   │   └───┘   └───┘                      ╱
           ┌───┐ ┌─┴─┐         ┌───┐                     ╱
    |q0>───┤ R ├─┤ X ├─────────┤ M ╞══                q3 ─────── q4
           └───┘ └───┘         └───┘


A SWAP-gate changes the mapping from logical qubits to physical qubits at a certain
point in the circuit, and thereby allows to solve mismatchings from the initial mapping.
The goal is to place SWAP-gates in the quantum circuit to fix the mismatches. The least
amount of SWAP-gates is preferred. In more advanced setups, also different factors can
be taken into account, like the fidelity of connections in the QPU.


State Space:
    The state space is described by a ``RoutingState`` with the following
    attributes:

    * `steps_done`: Number of steps done since the last reset.
    * `num_nodes`: Number of *physical* qubits.
    * `connection_graph`: A networkx representation of the connection graph.
    * `mapping`: Array of which the index represents a physical qubit, and the value a
      virtual qubit. This is updated after each swap.
    * `max_interaction_gates`: Maximum amount of gates allowed in the interaction 
      circuit, when a new interaction circuit is (randomly) generated.
    * `interaction_circuit`: An array of 2-tuples of integers, where every tuple
      represents a, not specified, gate acting on the two qubits labeled by the
      integers in the tuples.
    * `position`: The position in the original connection circuit.
    * `max_observation_reach`:  Caps the maximum amount of gates the agent can see ahead
      when making an observation.
    * `observation_booleans_flag`: If ``True`` a list called `boolean_flags` will be
      added to the observation space. The list `boolean_flags` has length
      `observation_reach` and containing Boolean values indicating whether the gates
      ahead can be executed.
    * `observation_connection_flag`: If ``True``, the connection_graph will be
        incorporated in the observation_space.
    * `swap_gates_inserted`: A list of 3-tuples of integers, to register which gates
        to insert and where. Every tuple (g, q1, q2) represents the insertion of a
        SWAP-gate acting on logical qubits q1 and q2 before gate g in the
        interaction_circuit.

Observation Space:
    The observation space is a ``qgym.spaces.Dict`` with 2-4 entries:

    * `interaction_gates_ahead`: Array with Boolean values for the upcoming connection
      gates in the quantum circuit.
    * `mapping`: The current state of the mapping.
    * (Optional)`connection_graph`: Adjacency matrix of the connection graph.
    * (Optional)`is_legal_surpass_booleans`: Array with boolean values stating wether a
      connection gate can be surpassed with the current mapping.

Action Space:
    A valid action is a tuple of integers  $(i,j,k)$. The integer $i$ indicates wether
    the agent wants to surpass the current gate and move on to the next gate. If $i$ is 
    0, then a SWAP gate is inserted at the current position between qubits $j$ and $k$.
    Only legal actions will be executed, an action is legal when:
    #. $i=1$ and the next gate can be executed.
    #. $i=0$ and physical qubit $j$ does not equal physical qubit $k$.
    #. $i=0$ and physical qubits $j$ and $k$ have a connection between them in the
      connection graph.


# TODO: create Examples


"""
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

import qgym.spaces
from qgym.envs.routing.routing_rewarders import BasicRewarder
from qgym.envs.routing.routing_state import RoutingState
from qgym.envs.routing.routing_visualiser import RoutingVisualiser
from qgym.templates import Environment, Rewarder
from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_bool,
    check_graph_is_valid_topology,
    check_instance,
    check_int,
)

Gridspecs = Union[List[Union[int, Iterable[int]]], Tuple[Union[int, Iterable[int]]]]


class Routing(
    Environment[
        Dict[str, Union[int, List[Tuple[int, int, int]], NDArray[np.int_]]],
        NDArray[np.int_],
    ],
):
    """RL environment for the routing problem of OpenQL."""

    def __init__(
        self,
        max_interaction_gates: int = 10,
        max_observation_reach: int = 5,
        observation_booleans_flag: bool = True,
        observation_connection_flag: bool = True,
        *,
        connection_graph: Optional[nx.Graph] = None,
        connection_graph_matrix: Optional[ArrayLike] = None,
        connection_grid_size: Optional[Gridspecs] = None,
        rewarder: Optional[Rewarder] = None,
    ) -> None:
        """Initialize the action space, observation space, and initial states.

        The supported render modes of this environment are "human" and "rgb_array".

        :param max_interaction_gates: Sets the maximum amount of gates in the
            `interaction_circuit`, when a new `interaction_circuit` is generated.
        :param max_observation_reach: Sets a cap on the maximum amount of gates the
            agent can see ahead when making an observation. When bigger than
            `max_interaction_gates` the agent will always see all gates ahead in an
            observation
        :param observation_booleans_flag: If ``True`` a list, of length
            observation_reach, containing booleans, indicating whether the gates ahead
            can be executed, will be added to the `observation_space`.
        :param observation_connection_flag: If ``True``, the connection_graph will be
            incorporated in the observation_space. Reason to set it ``False`` is:
            QPU-topology practically doesn't change a lot for one machine, hence an
            agent is typically trained for just one QPU-topology which can be learned
            implicitly by rewards and/or the booleans if they are shown, depending on
            the other flag above.
        :param connection_graph: ``networkx`` graph representation of the QPU topology.
            Each node represents a physical qubit and each node represents a connection
            in the QPU topology.
        :param connection_graph_matrix: Adjacency matrix representation of the QPU
            topology.
        :param connection_grid_size: Size of the connection graph when the connection
            graph has a grid topology. For more information on the allowed values and
            types, see ``networkx`` `grid_graph`_ documentation.
        :param rewarder: Rewarder to use for the environment. Must inherit from
            ``qgym.Rewarder``. If ``None`` (default), then ``BasicRewarder`` is used.

        .. _grid_graph: https://networkx.org/documentation/stable/reference/generated/
            networkx.generators.lattice.grid_graph.html#grid-graph
        """
        connection_graph = self._parse_connection_graph(
            connection_graph=connection_graph,
            connection_graph_matrix=connection_graph_matrix,
            connection_grid_size=connection_grid_size,
        )

        max_interaction_gates = check_int(
            max_interaction_gates, "max_interaction_gates", l_bound=0
        )
        max_observation_reach = check_int(
            max_observation_reach,
            "max_observation_reach",
            l_bound=0,
            u_bound=max_interaction_gates,
        )
        observation_booleans_flag = check_bool(
            observation_booleans_flag, "observation_booleans_flag", safe=False
        )
        observation_booleans_flag = check_bool(
            observation_connection_flag, "observation_connection_flag", safe=False
        )

        # Define internal attributes
        self._rewarder = self._parse_rewarder(rewarder)

        self._state = RoutingState(
            max_interaction_gates=max_interaction_gates,
            max_observation_reach=max_observation_reach,
            connection_graph=connection_graph,
            observation_booleans_flag=observation_booleans_flag,
            observation_connection_flag=observation_connection_flag,
        )
        self.observation_space = self._state.create_observation_space()

        # Define attributes defined in parent class
        self.action_space = qgym.spaces.MultiDiscrete(
            nvec=[self._state.n_qubits, self._state.n_qubits], rng=self.rng
        )

        self.metadata = {"render.modes": ["human", "rgb_array"]}

        self._visualiser = RoutingVisualiser(connection_graph)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        interaction_circuit: Optional[ArrayLike] = None,
        **_kwargs: Any,
    ) -> Union[
        Dict[str, Union[int, List[Tuple[int, int, int]], NDArray[np.int_]]],
        Tuple[
            Dict[str, Union[int, List[Tuple[int, int, int]], NDArray[np.int_]]],
            Dict[str, Any],
        ],
    ]:
        """Reset the state and set/create a new interaction circuit.

        To be used after an episode is finished.

        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call i.e., before any learning is done.
        :param return_info: Whether to receive debugging info. Default is ``False``.
        :param _kwargs: Additional options to configure the reset.
        :return: Initial observation and optionally debugging info.
        """
        # parse interaction circuit
        if interaction_circuit is not None:
            interaction_circuit = np.array(interaction_circuit)
            if (
                interaction_circuit.ndim != 2
                or interaction_circuit.shape[0] > self._state.max_interaction_gates
                or interaction_circuit.shape[1] != 2
            ):
                msg = "'interaction_circuit' should have be an ArrayLike with shape "
                msg + "(n_interactions,2), where n_interactions<=max_interaction_gates."
                raise ValueError(msg)
        # call super method for dealing with the general stuff
        return super().reset(
            seed=seed, return_info=return_info, interaction_circuit=interaction_circuit
        )

    @staticmethod
    def _parse_connection_graph(
        *,
        connection_graph: Any,
        connection_graph_matrix: Any,
        connection_grid_size: Any,
    ) -> nx.Graph:
        """Parse the user input (given in ``__init__``) to create a connection graph.

        :param connection_graph: ``networkx.Graph`` representation of the QPU topology.
        :param connection_graph_matrix: Adjacency matrix representation of the QPU
            topology
        :param connection_grid_size: Size of the connection graph when the topology is a
            grid.
        :raise ValueError: When `connection_graph`, `connection_graph_matrix` and
            `connection_grid_size` are all None.
        :return: Connection graph as a ``networkx.Graph``.
        """
        if connection_graph is not None:
            if connection_graph_matrix is not None:
                msg = "Both 'connection_graph' and 'connection_graph_matrix' were "
                msg += "given. Using 'connection_graph'."
                warnings.warn(msg)
            if connection_grid_size is not None:
                msg = "Both 'connection_graph' and 'connection_grid_size' were given. "
                msg += "Using 'connection_graph'."
                warnings.warn(msg)

            check_graph_is_valid_topology(connection_graph, "connection_graph")

            # deepcopy the graphs for safety
            return deepcopy(connection_graph)

        if connection_graph_matrix is not None:
            if connection_grid_size is not None:
                msg = "Both 'connection_graph_matrix' and 'connection_grid_size' were "
                msg += "given. Using 'connection_graph_matrix'."
                warnings.warn(msg)
            return Routing._parse_adjacency_matrix(connection_graph_matrix)
        if connection_grid_size is not None:
            # Generate connection grid graph
            return nx.grid_graph(connection_grid_size)

        msg = "No valid arguments for instantiation of the initial mapping environment "
        msg += "were provided."
        raise ValueError(msg)

    @staticmethod
    def _parse_rewarder(rewarder: Union[Rewarder, None]) -> Rewarder:
        """Parse the `rewarder` given by the user.

        :param rewarder: ``Rewarder`` to use for the environment. If ``None``, then the
            ``BasicRewarder`` with default settings is used.
        :return: Rewarder.
        """
        if rewarder is None:
            return BasicRewarder()
        check_instance(rewarder, "rewarder", Rewarder)
        return deepcopy(rewarder)

    @staticmethod
    def _parse_adjacency_matrix(connection_graph_matrix: ArrayLike) -> nx.Graph:
        """Parse a given connection graph adjacency matrix to its respective graph.

        :param connection_graph_matrix: adjacency matrix representation of the QPU
            topology.
        :raise TypeError: When the provided matrix is not a valid adjacency matrix.
        :return: Graph representation of the adjacency matrix.
        """
        connection_graph_matrix = check_adjacency_matrix(connection_graph_matrix)
        return nx.from_numpy_array(connection_graph_matrix)
