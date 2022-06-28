"""
Environment for training an RL agent on the scheduling problem of OpenQL.
"""
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.environment import Environment
from qgym.envs.scheduling_rewarders import BasicRewarder
from qgym.envs.scheduling_visualiser import SchedulingVisualiser
from qgym.utils import GateEncoder, RandomCircuitGenerator


class Scheduling(Environment):
    """
    RL environment for the scheduling problem.
    """

    def __init__(
        self, machine_properties: Union[Mapping[str, Any], str], max_gates: int = 200
    ) -> None:

        if isinstance(machine_properties, str):
            raise NotImplementedError(
                "Loading machine properties from files is not yet implemented."
            )

        n_qubits = machine_properties["qubit_number"]
        gate_encoder = GateEncoder().learn_gates(machine_properties["gates"])
        gate_cycle_length = gate_encoder.encode_gates(machine_properties["gates"])
        same_start = gate_encoder.encode_gates(
            machine_properties["machine_restrictions"]["same_start"]
        )
        not_in_same_cycle = gate_encoder.encode_gates(
            machine_properties["machine_restrictions"]["not_in_same_cycle"]
        )
        random_circuit_generator = RandomCircuitGenerator(
            n_qubits, max_gates, rng=self.rng
        )

        self._state = {
            "max_gates": max_gates,
            "n_qubits": n_qubits,
            "gate_encoder": gate_encoder,
            "random_circuit_generator": random_circuit_generator,
            "gate_cycle_length": gate_cycle_length,
            "same_start": same_start,
            "not_in_same_cycle": not_in_same_cycle,
        }

        self.reset()

        n_gate_names = gate_encoder.n_gates

        legal_actions_space = qgym.spaces.MultiBinary(max_gates, rng=self.rng)
        gate_names_space = qgym.spaces.MultiDiscrete(
            np.full(max_gates, n_gate_names + 1), rng=self.rng
        )
        acts_on_space = qgym.spaces.MultiDiscrete(
            np.full((2, max_gates), n_qubits + 1), rng=self.rng
        )
        scheduled_after_space = qgym.spaces.MultiDiscrete(
            np.full((n_qubits, max_gates), max_gates + 1), rng=self.rng
        )

        self.observation_space = qgym.spaces.Dict(
            rng=self.rng,
            legal_actions=legal_actions_space,
            gate_names=gate_names_space,
            acts_on=acts_on_space,
            scheduled_after=scheduled_after_space,
        )

        self.action_space = qgym.spaces.MultiDiscrete([max_gates, 2], rng=self.rng)

        self._rewarder = BasicRewarder()
        self._visualiser = SchedulingVisualiser(
            gate_encoder=gate_encoder,
            gate_cycle_length=gate_cycle_length,
            n_qubits=n_qubits,
        )

        self.metadata = {"render.modes": ["human"]}

    def _obtain_observation(self) -> Dict[str, NDArray[np.int_]]:
        """
        :return: Observation based on the current state.
        """
        return {
            "gate_names": self._state["gate_names"],
            "acts_on": self._state["acts_on"],
            "scheduled_after": self._state["scheduled_after"],
            "legal_actions": self._state["legal_actions"],
        }

    def _update_state(self, action: NDArray[np.int_]) -> None:
        """
        Update the state of this environment using the given action.

        :param action: First entry determines a gate to schedule, the second entry
            increases the cycle if nonzero.
        """
        # Increase the step number
        self._state["steps_done"] += 1

        # Increase the cycle if the action is given
        if action[1]:
            self._increment_cycle()

        # Schedule the gate if it is allowed
        gate_to_schedule = action[0]
        if self._state["legal_actions"][gate_to_schedule]:
            self._schedule_gate(gate_to_schedule)

    def _increment_cycle(self):
        """Increment the cycle and update the state accordingly"""
        self._state["cycle"] += 1

        # Reduce the amount of cycles each qubit is busy
        for idx, value in enumerate(self._state["busy"]):
            if value != 0:
                self._state["busy"][idx] -= 1

        # Exclude gates that should start at the same time
        while len(self._state["exclude_in_next_cycle"]) != 0:
            gate_to_exlude = self._state["exclude_in_next_cycle"].pop()
            self._exclude_gate(gate_to_exlude)

        # Decrease the number of cycles to exclude a gate and skip gates where the
        # cycle becomes 0 (as it no longer should be excluded)
        updated_excluded_gates = {}
        while len(self._state["excluded_gates"]) != 0:
            gate_intname, cycles = self._state["excluded_gates"].popitem()
            updated_cycles = cycles - 1
            if updated_cycles != 0:
                updated_excluded_gates[gate_intname] = updated_cycles
        self._state["excluded_gates"] = updated_excluded_gates

        self._update_legal_actions()

    def _schedule_gate(self, gate_idx):
        """Schedule a gate in the current cycle and update the state accordingly

        :param gate_idx: Index of the gate to schedule"""
        self._state["schedule"][gate_idx] = self._state["cycle"]
        gate_intname, control_qubit, target_qubit = self._state["encoded_circuit"][
            gate_idx
        ]
        self._state["busy"][control_qubit] = self._state["gate_cycle_length"][
            gate_intname
        ]
        self._state["busy"][target_qubit] = self._state["gate_cycle_length"][
            gate_intname
        ]

        if gate_intname in self._state["not_in_same_cycle"]:
            for gate_to_exlude in self._state["not_in_same_cycle"][gate_intname]:
                self._exclude_gate(gate_to_exlude)

        if gate_intname in self._state["same_start"]:
            self._state["exclude_in_next_cycle"].add(gate_intname)

        self._update_legal_actions()

    def _compute_reward(
        self,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        *_args: Any,
        **_kwargs: Any,
    ) -> float:
        """
        Asks the rewarder to compute a reward, given the current state.
        """
        return super()._compute_reward(
            old_state=old_state, action=action, new_state=self._state
        )

    def _is_done(self) -> bool:
        """
        :return: Boolean value stating whether we are in a final state.
        """
        return bool(
            (self._state["schedule"] != -1).sum() == len(self._state["encoded_circuit"])
        )

    def _obtain_info(self) -> Dict[Any, Any]:
        """
        :return: Optional debugging info for the current state.
        """
        return {
            "Steps done": self._state["steps_done"],
            "Cycle": self._state["cycle"],
            "Schedule": self._state["schedule"],
        }

    def reset(
        self,
        *,
        circuit: Optional[List[Tuple[str, int, int]]] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
    ) -> Union[
        Tuple[NDArray[np.int_], NDArray[np.int_]],
        Tuple[Tuple[NDArray[np.int_], NDArray[np.int_]], Dict[Any, Any]],
    ]:
        """
        Reset state, action space and step number and load a new (random) initial state.
        To be used after an episode is finished.

        :param circuit: Optional list of a circuit for the next episode, each entry in
            the list must containt the name, controll qubit and target qubit. If the
            gate has no controll, then these entries should be the same.
        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call, i.e. before any learning is done.
        :param return_info: Whether to receive debugging info.
        :param _kwargs: Additional options to configure the reset.
        :return: Initial observation and optional debugging info.
        """

        self._state["steps_done"] = 0
        self._state["cycle"] = 0

        # number of cycles that a qubit is still busy (zero if available)
        self._state["busy"] = np.zeros(self._state["n_qubits"], dtype=int)

        self._state["excluded_gates"] = {}
        self._state["exclude_in_next_cycle"] = set()

        if circuit is None:
            circuit = self._state["random_circuit_generator"].generate_circuit()

        self._state["encoded_circuit"] = self._state["gate_encoder"].encode_gates(
            circuit
        )
        self._state["schedule"] = np.full(len(circuit), -1, dtype=int)

        self._update_episode_constant_observations()
        self._update_legal_actions()

        # call super method for dealing with the general stuff
        return super().reset(seed=seed, return_info=return_info)

    def _update_episode_constant_observations(self) -> None:
        """
        Updates episode constant observations "gate_names", "acts_on" and
        "scheduled_after" based on the circuit of this episode.
        """
        circuit = self._state["encoded_circuit"]

        gate_names = np.full(
            self._state["max_gates"], self._state["gate_encoder"].n_gates, dtype=int
        )
        acts_on = np.full(
            (2, self._state["max_gates"]),
            self._state["gate_encoder"].n_gates,
            dtype=int,
        )
        scheduled_after = np.full(
            (self._state["n_qubits"], self._state["max_gates"]),
            self._state["max_gates"],
            dtype=int,
        )

        for idx, (gate_name, control_qubit, target_qubit) in enumerate(circuit):
            gate_names[idx] = gate_name
            acts_on[0, idx] = control_qubit
            acts_on[1, idx] = target_qubit

            # check the dependencies
            scheduled_after[0, idx] = self._find_next_gate(idx, control_qubit)
            if control_qubit != target_qubit:
                scheduled_after[1, idx] = self._find_next_gate(idx, target_qubit)

        self._state["gate_names"] = gate_names
        self._state["acts_on"] = acts_on
        self._state["scheduled_after"] = scheduled_after

    def _find_next_gate(self, start_idx, qubit):
        """Finds the next gate acting on the qubit, start searching from start_idx

        :param start_idx: index to start the search
        :param qubit: qubits to search on
        :return: index of the next gate acting on qubit"""
        for idx2 in range(start_idx + 1, len(self._state["encoded_circuit"])):
            control_qubit2 = self._state["encoded_circuit"][idx2][1]
            target_qubit2 = self._state["encoded_circuit"][idx2][2]
            if qubit == control_qubit2 or qubit == target_qubit2:
                return idx2
        return self._state["max_gates"]

    def _exclude_gate(self, gate_intname):
        gate_cyle_length = self._state["gate_cycle_length"][gate_intname]
        self._state["excluded_gates"][gate_intname] = gate_cyle_length

    def _update_legal_actions(self) -> None:
        """Checks which actions are legal based on the scheduled qubits, depedencies
        and in the futere also machine restrictions"""
        legal_actions = np.zeros(self._state["max_gates"], dtype=bool)
        for gate_idx, (gate_intname, control_qubit, target_qubit) in enumerate(
            self._state["encoded_circuit"]
        ):
            # Set all gates which have not been scheduled to True
            if self._state["schedule"][gate_idx] == -1:
                legal_actions[gate_idx] = True

            # Check if dependent gates are already scheduled
            for i in range(self._state["n_qubits"]):
                dependent_gate = self._state["scheduled_after"][i, gate_idx]
                if dependent_gate == self._state["max_gates"]:
                    continue

                if self._state["schedule"][dependent_gate] == -1:
                    legal_actions[gate_idx] = False

            # Check if the qubits are busy
            control_busy = self._state["busy"][control_qubit] != 0
            target_busy = self._state["busy"][target_qubit] != 0
            if control_busy or target_busy:
                legal_actions[gate_idx] = False

            # Check if gates should be excluded
            if gate_intname in self._state["excluded_gates"]:
                legal_actions[gate_idx] = False

        self._state["legal_actions"] = legal_actions

    def get_circuit(self, mode: str = "human"):
        """Return the quantum circuit of this episode

        :param mode: Choose from be 'human' or 'encoded'. Default is 'human'
        :return: human or encoded quantum circuit"""
        if not isinstance(mode, str):
            raise TypeError(f"mode must be of type str, but was {type(mode)}")

        if mode.lower() == "encoded":
            return deepcopy(self._state["encoded_circuit"])

        elif mode.lower() == "human":
            return self._state["gate_encoder"].decode_gates(
                self._state["encoded_circuit"]
            )

        else:
            raise ValueError(f"mode must be 'human' or 'encoded', but was {mode}")

    def render(self, mode: str = "human") -> bool:
        """
        Render the current state using pygame.
        :param mode: The mode to render with (default is 'human')
        """
        if mode not in self.metadata["render.modes"]:
            raise ValueError("The given render mode is not supported.")

        return self._visualiser.render(self._state)

    def close(self):
        """
        Close the screen used for rendering
        """
        self._visualiser.close()
