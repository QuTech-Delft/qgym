"""This module contains the :class:`RoutingState` class.

This :class:`RoutingState` represents the :class:`~qgym.templates.State` of the
:class:`~qgym.envs.Routing` environment.

Usage:
    >>> from qgym.envs.routing.routingstate import RoutingState
    >>> from qgym.generators import BasicInteractionGenerator
    >>> import networkx as nx
    >>> connection_graph = nx.grid_graph((3,3))
    >>> state = RoutingState(
    >>>             interaction_generator = BasicInteractionGenerator(9, 100),
    >>>             max_observation_reach = 20,
    >>>             connection_graph = connection_graph,
    >>>             observe_legal_surpasses = True,
    >>>             observe_connection_graph = False,
    >>>             )
"""

from __future__ import annotations

from collections import deque
from itertools import starmap
from typing import TYPE_CHECKING, Any, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.templates.state import State
from qgym.utils.input_parsing import has_fidelity

if TYPE_CHECKING:
    from qgym.generators.interaction import InteractionGenerator


# pylint: disable=too-many-instance-attributes


class RoutingState(State[dict[str, Union[NDArray[np.int_], NDArray[np.int8]]], int]):
    """The :class:`RoutingState` class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        interaction_generator: InteractionGenerator,
        max_observation_reach: int,
        connection_graph: nx.Graph,
        observe_legal_surpasses: bool,
        observe_connection_graph: bool,
    ) -> None:
        """Init of the ``RoutingState`` class.

        Args:
            interaction_generator: Interaction generator for generating interaction
                circuits. This generator is used to generate a new interaction circuit
                when :func:`RoutingState.reset` is called without an interaction
                circuit.
            max_observation_reach: Sets a cap on the maximum amount of gates the agent
                can see ahead when making an observation. When bigger than
                max_interaction_gates the agent will always see all gates ahead in an
                observation
            connection_graph: ``networkx`` graph representation of the QPU topology.
                Each node represents a physical qubit and each edge represents a
                connection in the QPU topology.
            observe_legal_surpasses: If ``True`` a boolean array of length
                max_observation_reach indicating whether the gates ahead can be
                executed, will be added to the `observation_space`.
            observe_connection_graph: If ``True``, the connection_graph will be
                incorporated in the `observation_space`. Reason to set it ``False`` is:
                QPU-topology doesn't change, hence an agent could infer the topology
                from the training data without needing to explicitly add it to the
                observations. This reduced the size `observation_space`.
        """
        self.steps_done = 0
        """Number of steps done since the last reset."""
        self.connection_graph = connection_graph
        """``networkx`` graph representation of the QPU topology. Each node represents a
        physical qubit and each edge represents a connection in the QPU topology.
        """
        self.edges = list(self.connection_graph.edges)
        """List of all the edges, used to decode given actions."""
        self.interaction_generator = interaction_generator
        """Sets the maximum amount of gates in the interaction_circuit, when a new
        interaction_circuit is generated.
        """
        self.interaction_circuit = np.pad(
            next(self.interaction_generator),
            ((0, max_observation_reach), (0, 0)),
            constant_values=self.n_qubits,
        )
        """An array of 2-tuples of integers, where every tuple represents a, not
        specified, gate acting on the two qubits labeled by the integers in the tuples.
        """
        self.mapping = np.arange(self.n_qubits, dtype=np.int_)
        """Array of which each index represents a logical qubit and each value
        represents a physical qubit.
        """
        self.position: int = 0
        """An integer representing before which gate in the interaction_circuit the
        agent currently is.
        """
        self.max_observation_reach = max_observation_reach
        """An integer that sets a cap on the maximum amount of gates the agent can see
        ahead when making an observation. When bigger than max_interaction_gates the
        agent will always see all gates ahead in an observation.
        """
        self.observe_legal_surpasses = observe_legal_surpasses

        if observe_connection_graph:
            if has_fidelity(connection_graph):
                self.connection_matrix = nx.to_numpy_array(
                    connection_graph, dtype=np.float64
                ).flatten()
            else:
                self.connection_matrix = nx.to_numpy_array(
                    connection_graph, dtype=np.bool_
                ).flatten()

        # Keep track of at what position which swap_gate is inserted
        self.swap_gates_inserted: deque[tuple[int, int, int]] = deque()
        """A deque of 3-tuples of integers, to register which gates to insert and where.
        Every tuple (g, q1, q2) represents the insertion of a SWAP-gate acting on
        physical qubits q1 and q2 before gate g in the interaction_circuit.
        """

    def reset(
        self,
        *,
        seed: int | None = None,
        interaction_circuit: NDArray[np.int_] | None = None,
        **_kwargs: Any,
    ) -> RoutingState:
        """Reset the state (in place) and load a new (random) initial state.

        To be used after an episode is finished.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call, i.e., before any learning is done.
            interaction_circuit: Optional 2D-Array of ints. Each int represents a qubit
                and a tuple of two ints represents an interaction between these qubits.
            _kwargs: Additional options to configure the reset.

        Returns:
            Self.
        """
        if seed is not None:
            self.seed(seed)

        if interaction_circuit is None:
            self.interaction_circuit = np.pad(
                next(self.interaction_generator),
                ((0, self.max_observation_reach), (0, 0)),
                constant_values=self.n_qubits,
            )
        else:
            interaction_circuit = np.array(interaction_circuit)
            if interaction_circuit.ndim != 2 or interaction_circuit.shape[1] != 2:
                msg = (
                    "'interaction_circuit' should have be an ArrayLike with shape "
                    "(n_interactions,2)."
                )
                raise ValueError(msg)
            self.interaction_circuit = np.pad(
                interaction_circuit,
                ((0, self.max_observation_reach), (0, 0)),
                constant_values=self.n_qubits,
            )

        # Reset position, counters
        self.position = 0
        self.steps_done = 0

        # resetting swap_gates_inserted and mapping
        self.swap_gates_inserted = deque()
        self.mapping = np.arange(self.n_qubits, dtype=np.int_)

        return self

    def obtain_info(
        self,
    ) -> dict[
        str, int | deque[tuple[int, int, int]] | NDArray[np.int_] | list[tuple[int]]
    ]:
        """Obtain additional information of the current state.

        Returns:
            Dictionary containing optional debugging info for the current state.
        """
        return {
            "Steps done": self.steps_done,
            "Position": self.position,
            "Interaction gates ahead": np.array(
                [
                    self.interaction_circuit[idx]
                    for idx in range(self.position, len(self.interaction_circuit))
                ]
            ),
            "Number of swaps inserted": len(self.swap_gates_inserted),
            "Swap gates inserted": self.swap_gates_inserted,
            "Action Encoding": self.edges,
        }

    def update_state(self, action: int) -> RoutingState:
        """Update the state (in place) of this environment using the given action.

        Args:
            action: Integer value in [0, n_connections]. Each value of 0 to
                n_connections-1 corresponds to placing a SWAP and this SWAP gate will be
                appended to the `swap_gates_inserted` ``deque``. The value of
                n_connections correspond to a surpass.

        Returns:
            Self.
        """
        # Increase the step number
        self.steps_done += 1

        # surpass current_gate if legal
        if action == self.n_connections:
            if self.is_legal_surpass(*self.interaction_circuit[self.position]):
                self.position += 1
            return self

        qubit1, qubit2 = self.edges[action]
        self._place_swap_gate(qubit1, qubit2)
        self._update_mapping(qubit1, qubit2)

        return self

    def create_observation_space(self) -> qgym.spaces.Dict:
        """Create the corresponding observation space.

        Returns:
            Observation space in the form of a :class:`~qgym.spaces.Dict` space
            containing:

            * :class:`~qgym.spaces.MultiDiscrete` space representing the interaction
              gates ahead of current position.
            * :class:`~qgym.spaces.MultiDiscrete` space representing the current mapping
              of logical onto physical qubits
        """
        interaction_gates_ahead = qgym.spaces.MultiDiscrete(
            np.full(2 * self.max_observation_reach, self.n_qubits + 1)
        )
        mapping = qgym.spaces.MultiDiscrete(np.full(self.n_qubits, self.n_qubits))

        observation_kwargs: dict[str, Any]
        observation_kwargs = {
            "interaction_gates_ahead": interaction_gates_ahead,
            "mapping": mapping,
        }

        if hasattr(self, "connection_matrix"):
            if has_fidelity(self.connection_graph):
                observation_kwargs["connection_graph"] = qgym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_qubits * self.n_qubits,),
                    dtype=self.connection_matrix.dtype,
                )
            else:
                observation_kwargs["connection_graph"] = qgym.spaces.MultiBinary(
                    self.n_qubits * self.n_qubits
                )

        if self.observe_legal_surpasses:
            observation_kwargs["is_legal_surpass"] = qgym.spaces.MultiBinary(
                self.max_observation_reach
            )

        return qgym.spaces.Dict(observation_kwargs)

    def obtain_observation(
        self,
    ) -> dict[str, NDArray[np.int_] | NDArray[np.int8]]:
        """Observe the current state.

        Returns:
            Observation based on the current state.
        """
        # construct interaction_gates_ahead
        gate_slice = slice(self.position, self.position + self.max_observation_reach)
        interaction_gates_ahead = self.interaction_circuit[gate_slice]

        observation: dict[str, NDArray[np.int_] | NDArray[np.int8]]
        observation = {
            "interaction_gates_ahead": interaction_gates_ahead.flatten(),
            "mapping": self.mapping,
        }

        if hasattr(self, "connection_matrix"):
            observation["connection_graph"] = self.connection_matrix

        if self.observe_legal_surpasses:
            is_legal_surpass = np.fromiter(
                iter=starmap(self.is_legal_surpass, interaction_gates_ahead),
                count=len(interaction_gates_ahead),
                dtype=np.int8,
            )
            observation["is_legal_surpass"] = is_legal_surpass

        return observation

    def is_done(self) -> bool:
        """Checks if the current state is in a final state.

        Returs: Boolean value stating whether we are in a final state.
        """
        return (
            self.position == len(self.interaction_circuit) - self.max_observation_reach
        )

    def _place_swap_gate(
        self,
        logical_qubit1: int,
        logical_qubit2: int,
    ) -> None:
        """Place a swap gate at the current position with the given logical qubits."""
        self.swap_gates_inserted.append((self.position, logical_qubit1, logical_qubit2))

    def is_legal_surpass(
        self,
        logical_qubit1: int,
        logical_qubit2: int,
    ) -> bool:
        """Checks whether a surpass of the current gate ahead is legal.

        Args:
            logical_qubit1: First qubit of the interaction.
            logical_qubit2: Second qubit of the logical interaction.

        Returns:
            A boolean value stating whether a connection gate with the two qubits can be
            executed with the current mapping and connection graph.
        """
        try:
            physical_connection = self.mapping[[logical_qubit1, logical_qubit2]]
        except IndexError:
            # The only logical qubits that are out of index, are those of padded gates.
            return True
        return physical_connection in self.connection_graph.edges

    def _update_mapping(
        self,
        logical_qubit1: int,
        logical_qubit2: int,
    ) -> None:
        """Updates mapping for a swap of two qubits."""
        physical_qubits = self.mapping[np.array([logical_qubit1, logical_qubit2])]
        self.mapping[np.array([logical_qubit2, logical_qubit1])] = physical_qubits

    @property
    def n_qubits(self) -> int:
        """Number of qubits in the `connection_graph`."""
        return int(self.connection_graph.number_of_nodes())

    @property
    def n_connections(self) -> int:
        """Number of connections in the `connection_graph`."""
        return len(self.edges)
