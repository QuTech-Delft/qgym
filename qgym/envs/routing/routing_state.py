# TODO: how to init -> test
# TODO: make observation space -> test

"""This module contains the ``RoutingState`` class.
This ``RoutingState``represents the ``State`` of the ``Routing`` environment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union, cast

import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.envs.routing.machine_properties import MachineProperties
from qgym.envs.routing.routing_dataclasses import (
    RoutingUtils,
    CircuitInfo,
)
from qgym.templates.state import State
from qgym.utils.random_circuit_generator import RandomCircuitGenerator


class SchedulingState(
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

    def __init__(
        self,
        *,
        machine_properties: MachineProperties,
        max_gates: int,
        dependency_depth: int,
        random_circuit_mode: str,
    ) -> None:

        self.steps_done = 0

        self.utils = RoutingUtils(
            random_circuit_generator=RandomCircuitGenerator(
                machine_properties.n_qubits, max_gates, rng=self.rng
            ),
            random_circuit_mode=random_circuit_mode,
            gate_encoder=machine_properties.encode(),
        )

        # Generate a random circuit
        circuit = self.utils.random_circuit_generator.generate_circuit(
            mode=self.utils.random_circuit_mode
        )

        self.circuit_info = CircuitInfo(
            encoded=self.utils.gate_encoder.encode_gates(circuit),
            names=np.empty(max_gates, dtype=int),
            acts_on=np.empty((2, max_gates), dtype=int),
            legal=np.empty(max_gates, dtype=bool),
            dependencies=np.empty((dependency_depth, max_gates), dtype=int),
        )

        self._update_dependencies()
        self._update_episode_constant_observations()
        self._update_legal_actions()
        
    def update_state(self, action: ActionT) -> State[ObservationT, ActionT]:
        """Update the state of this ``Environment`` using the given action.

        :param action: Action to be executed.
        """
        raise NotImplementedError

    def obtain_observation(self) -> ObservationT:
        """:return: Observation based on the current state."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        #when circuit including SWAPs can be implemented given the QPU connection graph 
        #TODO
        raise NotImplementedError

    def obtain_info(self) -> Dict[Any, Any]:
        """:return: Optional debugging info for the current state."""
        raise NotImplementedError

    def create_observation_space(self, observation_reach: int) -> Space:
        """Create the corresponding observation space."""
        #looking n gates ahead, showing for every gate whether the gate can be executed 
        #this can be indicated by an array of binary values:
        #return -> multibinary
        
        #TODO
        raise NotImplementedError