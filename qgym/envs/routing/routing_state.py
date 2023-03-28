# TODO: how to init -> test
# TODO: make observation space -> test

"""This module contains the ``RoutingState`` class.
This ``RoutingState``represents the ``State`` of the ``Routing`` environment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union, cast, Tuple

import numpy as np
from numpy.typing import NDArray

from qgym.custom_types import Gate

import qgym.spaces
from qgym.envs.scheduling.machine_properties import MachineProperties
from qgym.envs.routing.routing_dataclasses import (
    RoutingUtils,
    CircuitInfo,
)
from qgym.templates.state import State
from qgym.utils.random_circuit_generator import RandomCircuitGenerator


class RoutingState(
    State[Dict[str, Union[NDArray[np.int_], NDArray[np.bool_]]], NDArray[np.int_]]
):
    """The ``RoutingState`` class.
    :ivar machine_properties: ``MachineProperties`` object containing machine properties
        and limitations.
    :ivar utils: ``RoutingUtils`` dataclass with a random circuit generator and
        a gate encoder.
    :ivar gates: Dictionary with gate names as keys and ``GateInfo`` dataclasses as
        values.
    :ivar steps_done: Number of steps done since the last reset.
    :ivar busy: Used internally for the hardware limitations.
    """
    
    #TODO: how to initialize with a giving Initial_Mapping? NOT NEEDED in RoutingState

    def __init__(
        self,
        *,
        machine_properties: MachineProperties,
        max_interaction_gates: int,
        random_circuit_mode: str,
        max_observation_depth: int,
    ) -> None:
        self.steps_done = 0

        self.utils = RoutingUtils(
            random_circuit_generator=RandomCircuitGenerator(
                machine_properties.n_qubits, max_gates, rng=self.rng
            ),
            random_circuit_mode=random_circuit_mode,
            gate_encoder=machine_properties.encode(),
        )
        self.mapping = None #TODO: How do I get the Initial mapping?

        # Generate a random circuit
        circuit = self.utils.random_circuit_generator.generate_circuit(
            mode=self.utils.random_circuit_mode
        )

        #probably leave this out:
        # self.circuit_info = CircuitInfo(
        #     encoded=self.utils.gate_encoder.encode_gates(circuit),
        #     names=np.empty(max_gates, dtype=int),
        #     acts_on=np.empty((2, max_gates), dtype=int),
        #     legal=np.empty(max_gates, dtype=bool),
        #     dependencies=np.empty((dependency_depth, max_gates), dtype=int),
        # )

        self.observation_depth = max_observation_depth
        self.max_interaction_gates = max_interaction_gates
        
        # Generate circuit only containing interaction gates
        #TODO: think about keeping track of where in the circuit the interaction gate are.
        self.interaction_circuit = [gate for gate in circuit if not gate.q1 == gate.q2]
        self.position : int = 0 #position of agent within interaction circuit
        
        # Keep track of at what position which swap_gate is inserted
        self.swap_gates_inserted : List[(int, Gate)] = []
        
        self._update_dependencies()
        self._update_episode_constant_observations()
        self._update_legal_actions()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            circuit: Optional[List[Gate]] = None,
            **_kwargs: Any,
        ) -> SchedulingState:
            """Reset the state and load a new (random) initial state.

            To be used after an episode is finished.

            :param seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call, i.e., before any learning is done.
            :param circuit: Optional list of a circuit for the next episode, each entry in
                the list should be a ``Gate``. When a circuit is give, no random circuit
                will be generated.
            :param _kwargs: Additional options to configure the reset.
            :return: Self.
            """
    #TODO think about initial mapping with new circuit
    
    def update_state(self, action : NDArray[np.int_]) -> RoutingState:
        """Update the state of this environment using the given action.

        :param action: If action[0]==True a SWAP-gate applied to qubits action[1], 
        action[2] will be placed before the first observed gate, if False the first 
        observed gate will be surpassed.
        :return: Updated state.
        """
        # Increase the step number
        self.steps_done += 1
        
        #surpass current_gate
        if not action[1]:
            self.position += 1
            return self

        # Insert random swap-gate
        SWAP_gate_to_insert = (action[1], action[2])
        if self._is_legal[SWAP_gate_to_insert]:
            self._place_SWAP_gate(action[1], action[2])           
        
        return self

    def obtain_observation(self) -> ObservationT:
        """:return: Observation based on the current state."""
        #TODO: check pseudo-code below and improve! 
        observation_dict : Dict[bool] = None #dict with booleans of length observation_reach
        for idx in range(observation_reach):
           q1 = self.interaction_circuit[position + idx].q1
           q2 = self.interaction_circuit[position + idx].q2
           observation_dict[idx] = None #check (q1,q2) connection with current mapping
        raise NotImplementedError

    def is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        return self.position == self.len(self.interaction_circuit)

    def obtain_info(self) -> Dict[Any, Any]:
        """:return: Optional debugging info for the current state."""
        raise NotImplementedError

    def create_observation_space(self) -> Space:
        """Create the corresponding observation space.
        
        :returns: Observation space in the form of a ``qgym.spaces.Dict`` space
            containing:

            * ``qgym.spaces.MultiBinary`` space representing the legal actions. If
              the value at index $i$ determines if gate number $i$ can be scheduled
              or not.
            * ``qgym.spaces.MultiDiscrete`` space representing the integer encoded
              gate names.
            * ``qgym.spaces.MultiDiscrete`` space representing the interaction of
              each gate (q1 and q2).
            * ``qgym.spaces.MultiDiscrete`` space representing the first $n$ gates
              that must be scheduled before this gate.
        """  
        observation_reach = min(max_observation_reach, len(interaction_circuit)- self.position)


        observation_space = qgym.spaces.MultiDiscrete(
            np.full(2 * observation_reach, n_qubits + 1), rng=self.rng
        )
        return observation_space
    
    def _place_SWAP_gate(self, qubit1:int, qubit2:int) -> None:       
        #TODO: from collections import DeQueue for more efficient storage
        self.swap_gates_inserted.append((self.position, qubit1, qubit2))
        
        #TODO: update_mapping accordingly
        self.update_mapping(qubit1, qubit2)
        
    def _is_legal(self, SWAP_gate: Tuple[int, int]):
        #TODO:checks