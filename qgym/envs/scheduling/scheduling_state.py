"""This module contains the :class:`SchedulingState` class.

This :class:`SchedulingState` represents the :class:`~qgym.templates.State` of the
:class:`~qgym.envs.Scheduling` environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.envs.scheduling.scheduling_dataclasses import (
    CircuitInfo,
    GateInfo,
    SchedulingUtils,
)
from qgym.templates.state import State

if TYPE_CHECKING:
    from qgym.custom_types import Gate
    from qgym.envs.scheduling.machine_properties import MachineProperties
    from qgym.envs.scheduling.rulebook import CommutationRulebook
    from qgym.generators.circuit import CircuitGenerator


class SchedulingState(
    State[dict[str, Union[NDArray[np.int_], NDArray[np.int8]]], NDArray[np.int_]]
):
    """The :class:`SchedulingState` class."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        machine_properties: MachineProperties,
        max_gates: int,
        dependency_depth: int,
        circuit_generator: CircuitGenerator,
        rulebook: CommutationRulebook,
    ) -> None:
        """Init of the :class:`SchedulingState` class.

        Args:
            machine_properties: A :class:`~qgym.envs.scheduling.MachineProperties`
                object.
            max_gates: Maximum number of gates allowed in a circuit.
            dependency_depth: Number of dependencies given in the observation.
                Determines the shape of the `dependencies` observation, which has the
                shape (dependency_depth, max_gates).
            circuit_generator: Generator class for generating circuits for training.
            rulebook: :class:`~qgym.envs.scheduling.CommutationRulebook` describing the
                commutation rules.
        """
        self.steps_done = 0
        """Number of steps done since the last reset."""
        self.cycle = 0
        """Current 'machine' cycle."""
        self.machine_properties = machine_properties
        """:class:`~qgym.envs.scheduling.MachineProperties` class containing machine
        properties and limitations.
        """
        self.utils = SchedulingUtils(
            circuit_generator=circuit_generator,
            rulebook=rulebook,
            gate_encoder=machine_properties.encode(),
        )
        """:class:`~qgym.envs.scheduling.scheduling_dataclasses.SchedulingUtils`
        dataclass with a random circuit generator, commutation rulebook and a gate
        encoder.
        """

        # At the start no gates should be excluded
        gate_cycle_lengths = cast("dict[int, int]", machine_properties.gates)
        not_in_same_cycle = cast(
            "dict[int, set[int]]", machine_properties.not_in_same_cycle
        )
        self.gates = {
            gate_name: GateInfo(
                cycle_length=gate_cycle_lengths[gate_name],
                not_in_same_cycle=not_in_same_cycle[gate_name],
            )
            for gate_name in gate_cycle_lengths
        }
        """Dictionary with gate names as keys and
        :class:`~qgym.envs.scheduling.scheduling_dataclasses.GateInfo` dataclasses as
        values.
        """
        self.busy = np.zeros(machine_properties.n_qubits, dtype=int)
        """Amount of cycles that a qubit is still busy (zero if available). Used
        internally for the hardware limitations.
        """

        # Generate a circuit
        circuit = next(self.utils.circuit_generator)

        self.circuit_info = CircuitInfo(
            encoded=self.utils.gate_encoder.encode_gates(circuit),
            names=np.empty(max_gates, dtype=int),
            acts_on=np.empty((2, max_gates), dtype=int),
            legal=np.empty(max_gates, dtype=np.int8),
            dependencies=np.empty((dependency_depth, max_gates), dtype=int),
            schedule=np.full(len(circuit), -1, dtype=int),
            blocking_matrix=self.utils.rulebook.make_blocking_matrix(circuit),
        )
        """:class:`~qgym.envs.scheduling.scheduling_dataclasses.CircuitInfo`` dataclass
        containing the encoded circuit and attributes used to update the state.
        """

        self._update_dependencies()
        self._update_episode_constant_observations()
        self._update_legal_actions()

    def _update_dependencies(self) -> None:
        """Compute and update the dependencies array of the current state."""
        self.circuit_info.dependencies = np.zeros_like(self.circuit_info.dependencies)

        for gate_idx, blocking_row in enumerate(self.circuit_info.blocking_matrix):
            blocking_gates = blocking_row[gate_idx:].nonzero()[0]
            for depth in range(
                min(self.circuit_info.dependencies.shape[0], blocking_gates.shape[0])
            ):
                self.circuit_info.dependencies[depth, gate_idx] = blocking_gates[depth]

    def _update_episode_constant_observations(self) -> None:
        """Update episode constant observations `gate_names` and `acts_on`.

        Based on the circuit of the current episode.
        """
        self.circuit_info.names = np.zeros_like(self.circuit_info.names)
        self.circuit_info.acts_on = np.zeros_like(self.circuit_info.acts_on)

        for gate_idx, gate in enumerate(self.circuit_info.encoded):
            self.circuit_info.names[gate_idx] = gate.name
            self.circuit_info.acts_on[0, gate_idx] = gate.q1
            self.circuit_info.acts_on[1, gate_idx] = gate.q2

    def _update_legal_actions(self) -> None:
        """Check which actions are legal based on the scheduled qubits.

        An action is legal if the gate could be scheduled based on the machine
        properties and commutation rules.
        """
        self.circuit_info.legal = np.zeros_like(self.circuit_info.legal)
        for gate_idx, (gate_name, qubit1, qubit2) in enumerate(
            self.circuit_info.encoded
        ):
            # Set all gates, which have not been scheduled to True
            if self.circuit_info.schedule[gate_idx] == -1:
                self.circuit_info.legal[gate_idx] = True

            # Check if there is a non-scheduled dependent gate

            dependent_gates = self.circuit_info.dependencies[:, gate_idx]
            if np.count_nonzero(dependent_gates) > 0:
                self.circuit_info.legal[gate_idx] = False
                continue

            # Check if the qubits are busy
            if self.busy[qubit1] > 0 or self.busy[qubit2] > 0:
                self.circuit_info.legal[gate_idx] = False
                continue

            # Check if gates should be excluded
            if self.gates[gate_name].exclude > 0:
                self.circuit_info.legal[gate_idx] = False
                continue

    def create_observation_space(self) -> qgym.spaces.Dict:
        """Create the corresponding observation space.

        Returns:
            Observation space in the form of a :class:`~qgym.spaces.Dict` space
            containing:

            * :class:`~qgym.spaces.MultiBinary` space representing the legal actions. If
              the value at index $i$ determines if gate number $i$ can be scheduled
              or not.
            * :class:`~qgym.spaces.MultiDiscrete` space representing the integer encoded
              gate names.
            * :class:`~qgym.spaces.MultiDiscrete` space representing the interaction of
              each gate (q1 and q2).
            * :class:`~qgym.spaces.MultiDiscrete` space representing the first $n$ gates
              that must be scheduled before this gate.
        """
        max_gates = len(self.circuit_info.legal)
        n_gates = self.machine_properties.n_gates
        n_qubits = self.machine_properties.n_qubits
        dependency_depth = self.circuit_info.dependencies.shape[0]

        legal_actions_space = qgym.spaces.MultiBinary(max_gates, rng=self.rng)
        gate_names_space = qgym.spaces.MultiDiscrete(
            np.full(max_gates, n_gates + 1), rng=self.rng
        )
        acts_on_space = qgym.spaces.MultiDiscrete(
            np.full(2 * max_gates, n_qubits + 1), rng=self.rng
        )
        dependencies_space = qgym.spaces.MultiDiscrete(
            np.full(dependency_depth * max_gates, max_gates), rng=self.rng
        )

        return qgym.spaces.Dict(
            rng=self.rng,
            legal_actions=legal_actions_space,
            gate_names=gate_names_space,
            acts_on=acts_on_space,
            dependencies=dependencies_space,
        )

    def obtain_observation(
        self,
    ) -> dict[str, NDArray[np.int_] | NDArray[np.int8]]:
        """Obtain an observation based on the current state.

        Returns:
            Observation based on the current state.
        """
        return {
            "gate_names": self.circuit_info.names,
            "acts_on": self.circuit_info.acts_on.flatten(),
            "dependencies": self.circuit_info.dependencies.flatten(),
            "legal_actions": self.circuit_info.legal,
        }

    def is_done(self) -> bool:
        """Determine if the state is done or not.

        Returns:
            Boolean value stating whether we are in a final state.
        """
        return all(self.circuit_info.schedule != -1)

    def obtain_info(self) -> dict[str, int | NDArray[np.int_]]:
        """Obtain additional information.

        Returns:
            Optional debugging info for the current state.
        """
        return {
            "Steps done": self.steps_done,
            "Cycle": self.cycle,
            "Schedule": self.circuit_info.schedule,
        }

    def update_state(self, action: NDArray[np.int_]) -> SchedulingState:
        """Update the state of this environment using the given action.

        Args:
            action: First entry determines a gate to schedule, the second entry
                increases the cycle if nonzero.

        Returns:
            Self.
        """
        # Increase the step number
        self.steps_done += 1

        # Increase the cycle if the action is given
        if action[1]:
            self._increment_cycle()
            return self

        # Schedule the gate if it is allowed
        gate_to_schedule = action[0]
        if self.circuit_info.legal[gate_to_schedule]:
            self._schedule_gate(gate_to_schedule)
        return self

    def reset(
        self,
        *,
        seed: int | None = None,
        circuit: list[Gate] | None = None,
        **_kwargs: Any,
    ) -> SchedulingState:
        """Reset the state and load a new (random) initial state.

        To be used after an episode is finished.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call, i.e., before any learning is done.
            circuit: Optional list of a circuit for the next episode, each entry in the
                list should be a ``Gate``. When a circuit is give, no random circuit
                will be generated.
            _kwargs: Additional options to configure the reset.

        Returns:
            Self.
        """
        if seed is not None:
            self.seed(seed)

        # Reset counters
        self.steps_done = 0
        self.cycle = 0

        # Amount of cycles that a qubit is still busy (zero if available)
        self.busy = np.zeros_like(self.busy)

        # At the start no gates should be excluded
        for gate_info in self.gates.values():
            gate_info.reset()

        # Generate a circuit if None is given
        self.circuit_info.reset(circuit, self.utils)

        self._update_dependencies()
        self._update_episode_constant_observations()
        self._update_legal_actions()

        return self

    def _increment_cycle(self) -> None:
        """Increment the cycle and update the state accordingly."""
        self.cycle += 1

        # Reduce the amount of cycles each qubit is busy
        self.busy[self.busy > 0] -= 1

        # Exclude gates that should start at the same time
        for gate_name, gate_info in self.gates.items():
            if gate_info.exclude_next_cycle:
                gate_info.exclude_next_cycle = False
                self._exclude_gate(gate_name)

        # Decrease the amount of cycles to exclude a gate and skip gates where the
        # cycle becomes 0 (as it no longer should be excluded)
        for gate_info in self.gates.values():
            if gate_info.exclude > 0:
                gate_info.exclude -= 1

        self._update_legal_actions()

    def _exclude_gate(self, gate_name: int) -> None:
        """Exclude a gate from the 'legal_actions' for 'gate_cycle_length' cycles.

        Args:
            gate_name: integer encoding of the name of the gate.
        """
        gate_cycle_length = self.gates[gate_name].cycle_length
        self.gates[gate_name].exclude = gate_cycle_length

    def _schedule_gate(self, gate_idx: int) -> None:
        """Schedule a gate in the current cycle and update the state accordingly.

        Args:
            gate_idx: Index of the gate to schedule.
        """
        gate = self.circuit_info.encoded[gate_idx]

        # add the gate to the schedule
        self.circuit_info.schedule[gate_idx] = self.cycle

        self.busy[gate.q1] = self.gates[gate.name].cycle_length
        self.busy[gate.q2] = self.gates[gate.name].cycle_length

        for gate_to_exclude in self.gates[gate.name].not_in_same_cycle:
            self._exclude_gate(gate_to_exclude)

        if gate.name in self.machine_properties.same_start:
            self.gates[gate.name].exclude_next_cycle = True

        # Update "dependencies" observation
        self.circuit_info.blocking_matrix[:gate_idx, gate_idx] = False
        self._update_dependencies()
        self._update_legal_actions()
