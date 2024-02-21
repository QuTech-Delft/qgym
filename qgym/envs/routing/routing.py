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
    The state space is described by a :class:`~qgym.envs.routing.RoutingState` with the
    following attributes:

    * `steps_done`: Number of steps done since the last reset.
    * `num_nodes`: Number of *physical* qubits.
    * `connection_graph`: A networkx representation of the connection graph.
    * `edges`: List of edges of the connection graph used for decoding actions.
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
    * `observe_legal_surpasses`: If ``True`` a list called `boolean_flags` will be
      added to the observation space. The list `boolean_flags` has length
      `observation_reach` and containing Boolean values indicating whether the gates
      ahead can be executed.
    * `observe_connection_graph`: If ``True``, the connection_graph will be incorporated
      in the observation_space.
    * `swap_gates_inserted`: A list of 3-tuples of integers, to register which gates to
      insert and where. Every tuple (g, q1, q2) represents the insertion of a SWAP-gate
      acting on logical qubits q1 and q2 before gate g in the interaction_circuit.

Observation Space:
    The observation space is a :class:`~qgym.spaces.Dict` with 2-4 entries:

    * `interaction_gates_ahead`: Array with Boolean values for the upcoming connection
      gates in the quantum circuit.
    * `mapping`: The current state of the mapping.
    * (Optional) `connection_graph`: Adjacency matrix of the connection graph.
    * (Optional) `is_legal_surpass_booleans`: Array with boolean values stating whether
      a connection gate can be surpassed with the current mapping.

Action Space:
    A valid action is an integer in the domain [0, n_connections]. The values 0 to
    n_connections-1 represent an added SWAP gate. The value of n_connections indicates
    that the agents wants to surpass the current gate and move to the next gate.
    
    Illegal actions will not be executed. An action is considered illegal when the agent
    want to surpass a gate that cannot be executed with the current mapping.


# TODO: create Examples


"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

import qgym.spaces
from qgym.envs.routing.interaction_generation import (
    BasicInteractionGenerator,
    InteractionGenerator,
)
from qgym.envs.routing.routing_rewarders import BasicRewarder
from qgym.envs.routing.routing_state import RoutingState
from qgym.envs.routing.routing_visualiser import RoutingVisualiser
from qgym.templates import Environment, Rewarder
from qgym.utils.input_parsing import (
    parse_connection_graph,
    parse_rewarder,
    parse_visualiser,
)
from qgym.utils.input_validation import check_bool, check_instance, check_int

if TYPE_CHECKING:
    Gridspecs = (
        list[int] | list[Iterable[int]] | tuple[int, ...] | tuple[Iterable[int], ...]
    )


class Routing(Environment[Dict[str, NDArray[np.int_]], int]):
    """RL environment for the routing problem of OpenQL."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        interaction_generator: InteractionGenerator | None = None,
        max_observation_reach: int = 5,
        observe_legal_surpasses: bool = True,
        observe_connection_graph: bool = False,
        *,
        connection_graph: nx.Graph | None = None,
        connection_graph_matrix: ArrayLike | None = None,
        connection_grid_size: Gridspecs | None = None,
        rewarder: Rewarder | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the action space, observation space, and initial states.

        The supported render modes of this environment are ``"human"`` and
        ``"rgb_array"``.

        Args:
            max_interaction_gates: Sets the maximum amount of gates in the
                `interaction_circuit`, when a new `interaction_circuit` is generated.
            max_observation_reach: Sets a cap on the maximum amount of gates the agent
                can see ahead when making an observation. When bigger that
                `max_interaction_gates` the agent will always see all gates ahead in an
                observation
            observe_legal_surpasses: If ``True`` a boolean array of length
                `observation_reach` indicating whether the gates ahead can be executed,
                will be added to the `observation_space`.
            observe_connection_graph: If ``True``, the connection_graph will be
                incorporated in the observation_space. Reason to set it ``False`` is:
                QPU-topology practically doesn't change a lot for one machine, hence an
                agent is typically trained for just one QPU-topology which can be
                learned implicitly by rewards and/or the booleans if they are shown,
                depending on the other flag above. Default is ``False``.
            connection_graph: ``networkx`` graph representation of the QPU topology.
                Each node represents a physical qubit and each node represents a
                connection in the QPU topology.
            connection_graph_matrix: Adjacency matrix representation of the QPU
                topology.
            connection_grid_size: Size of the connection graph when the connection graph
                has a grid topology. For more information on the allowed values and
                types, see ``networkx`` `grid_graph`_ documentation.
            rewarder: Rewarder to use for the environment. Must inherit from
                :class:`~qgym.templates.Rewarder`. If ``None`` (default), then
                :class:`~qgym.envs,routing.BasicRewarder` is used.
            render_mode: If ``"human"`` open a ``pygame`` screen visualizing the step.
                If ``"rgb_array"``, return an RGB array encoding of the rendered frame
                on each render call.

        .. _grid_graph: https://networkx.org/documentation/stable/reference/generated/
            networkx.generators.lattice.grid_graph.html#grid-graph
        """
        # Check user input and parse it to a uniform format
        connection_graph = parse_connection_graph(
            connection_graph, connection_graph_matrix, connection_grid_size
        )

        if interaction_generator is None:
            interaction_generator = BasicInteractionGenerator(
                len(connection_graph), seed=self.rng
            )
        else:
            check_instance(
                interaction_generator, "interaction_generator", InteractionGenerator
            )
            if interaction_generator.finite:
                raise ValueError(
                    "'interaction_generator' should not be an infinite iterator"
                )
            interaction_generator = deepcopy(interaction_generator)

        max_observation_reach = check_int(
            max_observation_reach, "max_observation_reach", l_bound=1
        )
        observe_legal_surpasses = check_bool(
            observe_legal_surpasses, "observe_legal_surpasses", safe=False
        )
        observe_connection_graph = check_bool(
            observe_connection_graph, "observe_connection_graph", safe=False
        )

        # Define internal attributes
        self._rewarder = parse_rewarder(rewarder, BasicRewarder)

        self._state = RoutingState(
            interaction_generator=interaction_generator,
            max_observation_reach=max_observation_reach,
            connection_graph=connection_graph,
            observe_legal_surpasses=observe_legal_surpasses,
            observe_connection_graph=observe_connection_graph,
        )
        self.observation_space = self._state.create_observation_space()

        # Define attributes defined in parent class
        self.action_space = qgym.spaces.Discrete(
            self._state.n_connections + 1, rng=self.rng
        )

        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self._visualiser = parse_visualiser(
            render_mode, RoutingVisualiser, [connection_graph]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.int_]], dict[str, Any]]:
        r"""Reset the state and set/create a new interaction circuit.

        To be used after an episode is finished.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call i.e., before any learning is done.
            options: Mapping with keyword arguments with additional options for the
                reset. Keywords can be found in the description of
                :class:`~qgym.envs.routing.RoutingState`.\
                :class:`~qgym.envs.routing.RoutingState.reset()`.

        Returns:
            Initial observation and debugging info.
        """
        # call super method for dealing with the general stuff
        return super().reset(seed=seed, options=options)
