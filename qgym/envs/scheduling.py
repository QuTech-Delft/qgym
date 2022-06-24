"""
Environment for training an RL agent on the scheduling problem of OpenQL.
"""
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pygame
from numpy.typing import NDArray

import qgym.spaces
from qgym.environment import Environment
from qgym.envs.scheduling_rewarders import BasicRewarder
from qgym.utils import GateEncoder

# Define some colors used during rendering
BACKGROUND_COLOR = (150, 150, 150)  # Gray
TEXT_COLOR = (225, 225, 225)  # White
GATE_COLOR = (0, 0, 0)  # Black


class Scheduling(Environment):
    """
    RL environment for the scheduling problem.
    """

    def __init__(
        self, machine_properties: Union[Mapping[str, Any], str], max_gates: int = 200
    ) -> None:

        if isinstance(machine_properties, str):
            raise NotImplementedError(
                "Loading machine properties from files is not " "yet implemented."
            )

        n_qubits = machine_properties["qubit_number"]
        gate_encoder = GateEncoder().learn_gates(machine_properties["gates"])

        self._state = {
            "max_gates": max_gates,
            "n_qubits": n_qubits,
            "gate_encoder": gate_encoder,
            "gate_cycle_length": deepcopy(machine_properties["gates"]),
            "same_start": deepcopy(
                machine_properties["machine_restrictions"]["same_start"]
            ),
            "not_in_same_cycle": deepcopy(
                machine_properties["machine_restrictions"]["not_in_same_cycle"]
            ),
        }

        self.reset()

        n_gate_names = gate_encoder.n_gates

        legal_actions_space = qgym.spaces.MultiBinary(max_gates, rng=self.rng)
        gate_names_space = qgym.spaces.MultiDiscrete(
            [n_gate_names + 1] * max_gates, rng=self.rng
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

        self.metadata = {"render.modes": ["human"]}

        # Rendering data
        self.screen = None
        self.is_open = False
        self.screen_width = 1500
        self.screen_height = 800
        self.padding = 10

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
            self._state["cycle"] += 1

            # Reduce the amount of cycles each qubit is busy
            for idx, value in enumerate(self._state["busy"]):
                if value != 0:
                    self._state["busy"][idx] -= 1

            # Exclude gates that should start at the same time
            while len(self._state["exclude_in_next_cycle"]) != 0:
                gate_to_exlude = self._state["exclude_in_next_cycle"].pop()
                self._exclude_gate(gate_to_exlude)

            # Decrease the number of cycles to exclude a gate
            gates_to_pop = set()
            for gate_name, cycles in self._state["excluded_gates"].items():
                updated_cycles = cycles - 1
                if updated_cycles == 0:
                    gates_to_pop.add(gate_name)
                else:
                    self._state["excluded_gates"][gate_name] = updated_cycles

            # Remove gates from the exclusion list that should not be excluded anymore
            while len(gates_to_pop) != 0:
                gate = gates_to_pop.pop()
                self._state["excluded_gates"].pop(gate)

            self._update_legal_actions()

        # Schedule the gate if it is allowed
        gate_to_schedule = action[0]
        if self._state["legal_actions"][gate_to_schedule]:
            self._state["has_been_scheduled"][gate_to_schedule] = True
            self._state["schedule"][gate_to_schedule] = self._state["cycle"]
            gate_name, control_qubit, target_qubit = self._state["original_circuit"][
                gate_to_schedule
            ]
            self._state["busy"][control_qubit] = self._state["gate_cycle_length"][
                gate_name
            ]
            self._state["busy"][target_qubit] = self._state["gate_cycle_length"][
                gate_name
            ]

            if gate_name in self._state["not_in_same_cycle"]:
                for gate_to_exlude in self._state["not_in_same_cycle"][gate_name]:
                    self._exclude_gate(gate_to_exlude)

            if gate_name in self._state["same_start"]:
                self._state["exclude_in_next_cycle"].add(gate_name)

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
            self._state["has_been_scheduled"].sum()
            == len(self._state["encoded_circuit"]) + 1
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

        # max_gates is used as filler, so it is always scheduled
        self._state["has_been_scheduled"] = np.zeros(
            self._state["max_gates"] + 1, dtype=bool
        )
        self._state["has_been_scheduled"][self._state["max_gates"]] = True

        if circuit is None:
            circuit = self._make_random_circuit()

        self._state["original_circuit"] = circuit

        self._state["encoded_circuit"] = self._state["gate_encoder"].encode_gates(
            circuit
        )
        self._state["excluded_gates"] = {}
        self._state["exclude_in_next_cycle"] = set()

        self._update_episode_constant_observations()
        self._update_legal_actions()

        self._state["schedule"] = np.full(
            len(self._state["encoded_circuit"]), -1, dtype=int
        )

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

    def _exclude_gate(self, gate_name):
        gate_cyle_length = self._state["gate_cycle_length"][gate_name]
        self._state["excluded_gates"][gate_name] = gate_cyle_length

    def _make_random_circuit(
        self, n_gates: Union[str, int] = "random"
    ) -> List[Tuple[str, int, int]]:
        """
        Make a random circuit with prep, measure, x, y, z, and cnot operations

        :param n_gates: If "random", then a circuit of random length will be made, if
            an int a circuit of length min(n_gates, max_gates) will be made.
        :return: A randomly generated circuit
        """
        circuit = []
        if n_gates.lower().strip() == "random":
            n_gates = self.rng.integers(
                low=self._state["n_qubits"],
                high=self._state["max_gates"],
                endpoint=True,
            )
        else:
            n_gates = min(n_gates, self._state["max_gates"])

        for qubit in range(self._state["n_qubits"]):
            circuit.append(("prep", qubit, qubit))

        gates = ["x", "y", "z", "cnot", "measure"]
        p = [0.16, 0.16, 0.16, 0.5, 0.02]
        for _ in range(n_gates - self._state["n_qubits"]):
            gate = self.rng.choice(gates, p=p)
            if gate == "cnot":
                control_qubit, target_qubit = self.rng.choice(
                    np.arange(self._state["n_qubits"]), size=2, replace=False
                )
                circuit.append((gate, control_qubit, target_qubit))
            else:
                qubit = self.rng.integers(self._state["n_qubits"])
                circuit.append((gate, qubit, qubit))

        return circuit

    def _update_legal_actions(self) -> None:
        """
        Checks which actions are legal based on the scheduled qubits and depedencies
        (and in the futere also machine restrictions)
        """
        legal_actions = np.zeros(self._state["max_gates"], dtype=bool)
        for gate_idx, (gate_name, control_qubit, target_qubit) in enumerate(
            self._state["original_circuit"]
        ):
            if not self._state["has_been_scheduled"][gate_idx]:
                legal_actions[gate_idx] = True

            # Check if dependent gates are already scheduled
            for i in range(self._state["n_qubits"]):
                dependent_gate = self._state["scheduled_after"][i, gate_idx]
                if not self._state["has_been_scheduled"][dependent_gate]:
                    legal_actions[gate_idx] = False

            # Check if the qubits are busy
            control_busy = self._state["busy"][control_qubit] != 0
            target_busy = self._state["busy"][target_qubit] != 0
            if control_busy or target_busy:
                legal_actions[gate_idx] = False

            # Check if gates should be excluded
            if gate_name in self._state["excluded_gates"]:
                legal_actions[gate_idx] = False

        self._state["legal_actions"] = legal_actions

    def render(self, mode: str = "human") -> bool:
        """
        Render the current state using pygame.
        :param mode: The mode to render with (default is 'human')
        """
        if mode not in self.metadata["render.modes"]:
            raise ValueError("The given render mode is not supported.")

        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Scheduling Environment")
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 12)
            self.is_open = True

        pygame.time.delay(10)

        self.screen.fill(BACKGROUND_COLOR)

        for gate_idx in range(len(self._state["encoded_circuit"])):
            if self._state["has_been_scheduled"][gate_idx]:
                self._draw_scheduled_gate(gate_idx)

        pygame.event.pump()
        pygame.display.flip()

        return self.is_open

    def _draw_scheduled_gate(self, gate_idx: int) -> None:

        gate_name, control_qubit, target_qubit = self._state["original_circuit"][
            gate_idx
        ]
        scheduled_cycle = self._state["schedule"][gate_idx]

        self._draw_gate_block(gate_name, control_qubit, scheduled_cycle)

        if control_qubit != target_qubit:
            self._draw_gate_block(gate_name, target_qubit, scheduled_cycle)

    def _draw_gate_block(
        self, gate_name: str, qubit: int, scheduled_cycle: int
    ) -> None:
        cycle_width = self.screen_width / (self._state["cycle"] + 10)
        box_width = cycle_width * self._state["gate_cycle_length"][gate_name]
        box_height = self.screen_height / self._state["n_qubits"]

        gate_box = pygame.Rect(0, 0, box_width, box_height)
        box_x = self.screen_width - scheduled_cycle * cycle_width
        box_y = self.screen_height - qubit * box_height
        gate_box.bottomright = (box_x, box_y)

        pygame.draw.rect(self.screen, GATE_COLOR, gate_box)

        text = self.font.render(gate_name.upper(), True, TEXT_COLOR)
        text_postition = text.get_rect(center=gate_box.center)
        self.screen.blit(text, text_postition)

    def close(self):
        """
        Closed the screen used for rendering
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()
            self.is_open = False
            self.screen = None
