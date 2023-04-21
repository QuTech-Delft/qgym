"""This module contains the ``RoutingState`` class.
This ``RoutingState``represents the ``State`` of the ``Routing`` environment.

Usage:
    >>> from qgym.envs.routing.routingstate import RoutingState
    >>> import networkx as nx
    >>> connection_graph = nx.grid_graph((3,3))
    >>> state = TODO
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.custom_types import Gate
from qgym.templates.state import State

class RoutingState(
    State[Dict[str, Union[NDArray[np.int_], NDArray[np.bool_]]], NDArray[np.int_]]
):
    """The ``RoutingState`` class.
    :ivar steps_done: Number of steps done since the last reset.
    :ivar connection_graph: ``networkx`` graph representation of the QPU topology.
            Each node represents a physical qubit and each edge represents a connection
            in the QPU topology.
    :ivar n_qubits: Number of qubits in the connection_graph
    :ivar max_interaction_gates: Sets the maximum amount of gates in the
            interaction_circuit, when a new interaction_circuit is generated.
    :ivar interaction_circuit: A list of 2-tuples of integers, where every tuple
        represents a, not specified, gate acting on the two qubits labeled by the
        integers in the tuples.
    :ivar current_mapping: List of which the index represents a physical qubit, and the
        value a logical qubit.
    :ivar position: An integer representing the before which gate in the
        interaction_circuit the agent currently is.
    :ivar max_observation_reach: An integer that sets a cap on the maximum amount of
        gates the agent can see ahead when making an observation. When bigger than
        max_interaction_gates the agent will always see all gates ahead in an
        observation.
    :ivar observation_reach: An integer representing the current amount of gates the
        agent can see when making an observation.
    :ivar swap_gates_inserted: A list of 3-tuples of integers, to register which gates
        to insert and where. Every tuple (g, q1, q2) represents the insertion of a
        SWAP-gate acting on qubits q1 and q2 before gate g in the interaction_circuit.
    """

    def __init__(
        self,
        *,
        max_interaction_gates: int,
        max_observation_reach: int,
        connection_graph: nx.Graph,
        observation_booleans_flag: bool,
        observation_connection_flag: bool,
    ) -> None:
        """Init of the ``RoutingState`` class.

        :param max_interaction_gates: Sets the maximum amount of gates in the
            interaction_circuit, when a new interaction_circuit is generated.
        :param max_observation_reach: Sets a cap on the maximum amount of gates the
            agent can see ahead when making an observation. When bigger than
            max_interaction_gates the agent will always see all gates ahead in an
            observation
        :param connection_graph: ``networkx`` graph representation of the QPU topology.
            Each node represents a physical qubit and each edge represents a connection
            in the QPU topology.
        :param observation_booleans_flag: If ``True`` a list, of length
        observation_reach, containing booleans, indicating whether the gates ahead can
        be executed, will be added to the observation_space.
        :param observation_connection_flag: If ``True``, the connection_graph will be
        incorporated in the observation_space. Reason to set it False is: QPU-topology
        practically doesn't change a lot for one machine, hence an agent is typically
        trained for just one QPU-topology which can be learned implicitly by rewards
        and/or the booleans if they are shown, depending on the other flag above.
        """
        self.steps_done: int = 0

        # topology
        self.connection_graph = connection_graph
        self.n_qubits: int = self.connection_graph.number_of_nodes()

        # interaction circuit + mapping
        self.max_interaction_gates = max_interaction_gates
        self.number_of_gates = int(self.rng.choice(range(1, max_interaction_gates + 1)))
        self.interaction_circuit: List[Tuple[int, int]]
        self.interaction_circuit = self.generate_random_interaction_circuit(self.number_of_gates)
        self.current_mapping = np.arange(self.n_qubits, dtype=np.uint8) 

        # Observation attributes
        self.position: int = 0
        self.max_observation_reach = int(
            min(max_observation_reach, len(self.interaction_circuit))
        )
        self.observation_reach = self.max_observation_reach
        self.observation_booleans_flag = observation_booleans_flag
        self.observation_connection_flag = observation_connection_flag

        # Keep track of at what position which swap_gate is inserted
        self.swap_gates_inserted: List[Tuple[int, int, int]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        interaction_circuit: Optional[Tuple[int, int]] = None,
        **_kwargs: Any,
    ) -> RoutingState:
        """Reset the state and load a new (random) initial state.

        To be used after an episode is finished.

        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call, i.e., before any learning is done.
        :param circuit: Optional list of tuples of ints that the interaction gates
            via the qubits the gates are acting on.
        :param _kwargs: Additional options to configure the reset.
        :return: Self.
        """
        if seed is not None:
            self.seed(seed)

        # Reset counters
        self.steps_done = 0

        if interaction_circuit is None:
            self.number_of_gates = int(
                self.rng.choice(range(1, self.max_interaction_gates + 1))
            )
            self.interaction_circuit = self.generate_random_interaction_circuit(
                self.number_of_gates
            )
        else:
            self.interaction_circuit = interaction_circuit

        # resetting swap_gates_inserted and mapping
        self.swap_gates_inserted = []
        self.mapping = [idx for idx in range(self.n_qubits)]

    def update_state(self, action: NDArray[np.int_]) -> RoutingState:
        """Update the state of this environment using the given action.

        :param action: If action[0]==0 a SWAP-gate applied to qubits action[1],
            action[2] will be registered in the swap_gates_inserted-list at the current
            position, if action[0]==1 the first observed gate will be surpassed.
        :return: self
        """
        # Increase the step number
        self.steps_done += 1

        # surpass current_gate if legal
        if action[0] == 1 and self._can_be_executed(
            self.interaction_circuit[self.position][0],
            self.interaction_circuit[self.position][1],
        ):
            self.position += 1
            # update observation reach
            if len(self.interaction_circuit) - self.position < self.observation_reach:
                self.observation_reach -= 1

        # elif insert random swap-gate if legal
        elif action[0] == 0 and self._is_legal_swap(action[1], action[2]):
            self._place_swap_gate(action[1], action[2])
            self._update_mapping(action[1], action[2])

        return self

    def obtain_observation(
        self,
    ) -> Dict[str, NDArray[np.int_]]:
        """:return: Observation based on the current state."""
        # TODO: check for efficient slicing!
        interaction_gates_ahead = list(
            itertools.chain(*self.interaction_circuit[-self.observation_reach :])
        )
        if self.observation_reach < self.max_observation_reach:
            difference = self.max_observation_reach - self.observation_reach
            interaction_gates_ahead += [self.n_qubits] * difference

        # TODO: Do we also want to show the topology in the observation?
        #   If so we could make use of the graps-dictionary storage format used in
        #   inital_mapping_state.
        return {
            "interaction_gates_ahead": interaction_gates_ahead,
            "current_mapping": self.current_mapping,
        }

    def is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        # self.observation_reach==0
        return self.position == self.len(self.interaction_circuit)

    def create_observation_space(self) -> Space:
        """Create the corresponding observation space.

        :returns: Observation space in the form of a ``qgym.spaces.Dict`` space
            containing:

            * ``qgym.spaces.MultiDiscrete`` space representing the interaction gates
                ahead of current position.
            * ``qgym.spaces.MultiDiscrete`` space representing the current mapping of
                logical onto physical qubits
        """
        interaction_gates_ahead = qgym.spaces.MultiDiscrete(
            np.full(2 * self.max_observation_reach, self.n_qubits)
        )
        current_mapping = qgym.spaces.MultiDiscrete(
            np.full(self.n_qubits, self.n_qubits)
        )
        # TODO: implement optional extension of observation_space based on flags.
        if not self.observation_connection_flag and not self.observation_booleans_flag:
            observation_space = qgym.spaces.Dict(
                interaction_gates_ahead=interaction_gates_ahead,
                current_mapping=current_mapping,
            )
        elif self.observation_connection_flag and not self.observation_booleans_flag:
            # TODO: implement.
            pass
        elif not self.observation_connection_flag and self.observation_booleans_flag:
            # TODO: implement.
            pass
        elif self.observation_connection_flag and self.observation_booleans_flag:
            # TODO: implement.
            pass
        return observation_space

    def _place_swap_gate(self, qubit1: int, qubit2: int) -> None:
        # TODO: STORAGE EFFICIENCY: from collections import DeQueue
        self.swap_gates_inserted.append((self.position, qubit1, qubit2))

    def _is_legal_swap(self, swap_qubit1: int, swap_qubit2: int):
        return (swap_qubit1 is not swap_qubit2) and (
            (swap_qubit1, swap_qubit2) in self.connection_graph.edges
        )

    def _can_be_executed(self, gate_qubit1: int, gate_qubit2: int):
        logical_gate_qubit1 = self.current_mapping[gate_qubit1]
        logical_gate_qubit2 = self.current_mapping[gate_qubit2]
        return (logical_gate_qubit1, logical_gate_qubit2) in self.connection_graph.edges

    def _update_mapping(self, qubit1: int, qubit2: int) -> None:
        logical1 = self.current_mapping[qubit1]
        logical2 = self.current_mapping[qubit2]
        self.current_mapping[qubit1] = logical2
        self.current_mapping[qubit2] = logical1

    def generate_random_interaction_circuit(
        self, n_gates: Union[str, int] = "random"
    ) -> List[Tuple[int, int]]:
        """Generate a random interaction circuit.

        :param n_gates: If "random", then a circuit of random length will be made. If
            an ``int`` is given, a circuit of length ``min(n_gates, max_gates)`` will
            be made.
        :return: A randomly generated interaction circuit.
        """
        n_gates = self._parse_n_gates(n_gates)

        circuit: List[Tuple[int, int]] = [0] * n_gates
        for idx in range(n_gates):
            qubit1, qubit2 = self.rng.choice(
                np.arange(self.n_qubits), size=2, replace=False
            )
            circuit[idx] = (qubit1, qubit2)

        return circuit

    def _parse_n_gates(self, n_gates: Union[int, str]) -> int:
        """Parse `n_gates`.

        :param n_gates: If n_gates is "random", generate a number between 1 and
            `max_gates`. If n_gates is an ``int``, return the minimum of `n_gates` and
            `max_gates`.
        """
        if isinstance(n_gates, str):
            if n_gates.lower().strip() == "random":
                return self.rng.integers(
                    self.n_qubits, self.max_interaction_gates, endpoint=True
                )

            raise ValueError(f"Unknown flag {n_gates}, choose from 'random'.")

        if isinstance(n_gates, int):
            return min(n_gates, self.max_interaction_gates)

        msg = f"n_gates should be of type int or str, but was of type {type(n_gates)}."
        raise ValueError(msg)
