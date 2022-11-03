r"""This module contains an environment for training an RL agent on the quantum
operation scheduling problem of OpenQL. The quantum operations scheduling problem is
aimed at finding the shortest possible schedules of operations defined by a **quantum
circuit**, whilst taking **hardware constraints** and **commutation rules** into
account.

Circuits:
    The quantum operations scheduling problem consist of scheduling the operations
    described by a quantum circuit. For the ``Scheduling`` environment in this module, a
    quantum circuit is defined as a list of gates, where a gate is a ``namedtuple`` with
    a 'name', 'q1' and 'q2'. For example, a c-not with control qubit 2 and target qubit
    5 would be ``Gate("cnot", 2, 5)`` and an X gate acting on qubit 4 would be
    ``Gate("x", 4, 4)``. In this representation, the circuit below can be created by:

    >>> from qgym.custom_types import Gate
    >>> circuit = [Gate("x", 1, 1), Gate("cnot", 0, 1), Gate("x", 0, 0), Gate("h",1,1)]

    .. code-block:: console

                  QUANTUM CIRCUIT
               ┌───┐   ┌───┐   ┌───┐
        |q1>───┤ Y ├───┤ X ├───┤ H ├──
               └───┘   └─┬─┘   └───┘
                         │     ┌───┐
        |q0>─────────────┴─────┤ X ├──
                               └───┘


Hardware specifications:
    Different operations defined by a quantum circuit have different operation times.
    These operation times are defined in the unit of machine cycles. For example, an X
    gate could take 2 machine cycles, whereas a measurement could take 15 cycles. These
    different operation times must be taken into account, because the hardware (the
    quantum computers) cannot perform multiple operations on the same qubit at the same
    time.

    Finding an optimal schedule with these restriction is already a difficult task.
    However, to complicate matters there can be more limitations defined by the
    hardware. To be more concrete, the ``Schdeuling`` environment can take two more
    types of limitations into account:

    #. Some gates must start at the same time, or must wait till the previous one is
       finished. This is typically the case for measurements.
    #. Some gates can't be executed in the same cycle. This for example happens often
       with X, Y and Z gates.

    The goal of the quantum operations scheduling problem is to find an optimal
    schedule, which takes these hardware limitations into account, as well as some
    optional commutation rules.

State space:
    The state space is described by a dictionary with the following structure:

    * ``max_gates``: Maximum circuit length allowed in the environment.
    * ``n_qubits``: Number of physical qubits of the hardware.
    * ``gate_cycle_length``: Mapping of integer encoded gates and their respective cycle
      length.
    * ``same_start``: Set of integer encoded gates that should start in the same cycle.
    * ``not_in_same_cycle``: Mapping of integer encoded gates that can't start in the
      same cycle.
    * ``steps_done``: Number of steps done since the last reset.
    * ``cycle``: Current 'machine' cycle.
    * ``busy``: Used internally for the hardware limitations.
    * ``dependencies``: Shows the first $n$ gates that must be scheduled before this
      gate.
    * ``encoded_circuit``: Integer encoded representation of the circuit to schedule.
    * ``gate_names``: Gate names of the encoded circuit.
    * ``acts_on``: q1 and q2 of each gate.
    * ``legal_actions``: Array of legal actions. If the value at index $i$ determines if
      gate number $i$ can be scheduled or not. This array is based on the machine
      restrictions and commutation rules.

Observation space:
    Each element in the observation space is a dictionary with 4 entries:

    * ``gate_names``: Gate names of the encoded circuit.
    * ``acts_on``: q1 and q2 of each gate.
    * ``dependencies``: Shows the first $n$ gates that must be scheduled before this
      gate.
    * ``legal_actions``: List of legal actions. If the value at index $i$ determines if
      gate number $i$ can be scheduled or not.

Action Space:
    Performing a quantum operation takes a certain amount of time, which is measured in
    (machine) cycles. Therefore, this environment aims to produce a schedule in terms
    of cycles. To do this, the environment schedules the circuit from right to left,
    so cycle zero in the schedule is the last operation to be performed in that
    schedule. Based on this idea, a valid action is then a tuple of an integer
    $i\in\{0,1,\ldots,max\_gates\}$ and a binary value $j$. The integer value $i$
    schedules gate $i$ in the current cycle, and the binary value $j$ increment the
    cycle number.

    If the action is illegal (for example when we try to schedule a gate that has been
    scheduled already), then the environment will do nothing, and the rewarder should
    give a penalty for this.

Example 1:
    In this example we want to create an environment for a machine with the following
    properties:

    * The machine has two qubits.
    * The machine supports the X, Y, C-NOT and Measure gates.
    * Multiple Measure gates should start in the same cycle, or wait till the previous
      one is done.
    * The X and Y gates cannot be performed in the same cycle.

    The code block below shows how a ``Scheduling`` environment for such a machine, were
    the environment only allows for circuit with a maximum length of 5:

    .. code-block:: python

        from qgym.envs.scheduling import Scheduling, MachineProperties

        # Set up the hardware specifications
        hardware_spec = MachineProperties(n_qubits=2)
        hardware_spec.add_gates({"x": 2, "y": 2, "cnot": 4, "measure": 10})
        hardware_spec.add_same_start(["measure"])
        hardware_spec.add_not_in_same_cycle([("x", "y")])

        # Initialize the environment
        env = Scheduling(hardware_spec, max_gates=5, random_circuit_mode="workshop")


Example 2:
    This example shows how to add a specified commutation rulebook, which can contain
    costum commutation rules. For more information on implementing custom cummutation
    rules, see the documentation of ``CommutationRulebook``. This example uses the same
    machine properties as example 1. Costrum commutation rules can be added as shown in
    the code block below:

    .. code-block:: python

        from qgym.envs.scheduling import Scheduling, MachineProperties
        from qgym.envs.scheduling.rulebook import CommutationRulebook

        # Set up the hardware specifications
        hardware_spec = MachineProperties(n_qubits=2)
        hardware_spec.add_gates({"x": 2, "y": 2, "cnot": 4, "measure": 10})
        hardware_spec.add_same_start(["measure"])
        hardware_spec.add_not_in_same_cycle([("x", "y")])

        # Setup the rulebook
        rulebook = CommutationRulebook()

        # Initialize the environment
        env_com = Scheduling(
                            hardware_spec,
                            max_gates=5,
                            rulebook=rulebook,
                            random_circuit_mode="workshop",
                        )


"""
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym import Rewarder
from qgym.custom_types import Gate
from qgym.environment import Environment
from qgym.envs.scheduling.machine_properties import MachineProperties
from qgym.envs.scheduling.rulebook import CommutationRulebook
from qgym.envs.scheduling.scheduling_rewarders import BasicRewarder
from qgym.envs.scheduling.scheduling_visualiser import SchedulingVisualiser
from qgym.utils import RandomCircuitGenerator
from qgym.utils.input_validation import check_instance, check_int, check_string


class Scheduling(Environment):
    """RL environment for the scheduling problem."""

    def __init__(
        self,
        machine_properties: Union[Mapping[str, Any], str, MachineProperties],
        *,
        max_gates: int = 200,
        dependency_depth: int = 1,
        random_circuit_mode: str = "default",
        rulebook: Optional[CommutationRulebook] = None,
        rewarder: Optional[Rewarder] = None,
    ) -> None:
        """Initialize the action space, observation space, and initial states for the
        scheduling environment.

        :param machine_properties: A ``MachineProperties`` object, a ``Mapping``
            containing machine properties or a string with a filename for a file
            containing the machine properties.
        :param max_gates: Maximum number of gates allowed in a circuit. Defaults to 200.
        :param dependency_depth: Number of dependencies given in the observation.
            Determines the shape of the `dependencies` observation, which has the shape
            (dependency_depth, max_gates). Defaults to 1.
        :random_circuit_mode: Mode for the random circuit generator. The mode can be
            'default' or 'workshop'. Defaults to 'default'.
        :param rulebook: ``CommutationRulebook`` describing the commutation rules. If
            ``None`` (default) is given, a default ``CommutationRulebook`` will be used.
            (See ``CommutationRulebook`` for more info on the default rules.)
        :param rewarder: Rewarder to use for the environment. If ``None`` (default),
            then a default ``BasicRewarder`` is used.
        """
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "random_circuit.mode": ["default", "workshop"],
        }

        machine_properties = self._parse_machine_properties(machine_properties)
        max_gates = check_int(max_gates, "max_gates", l_bound=1)
        self._dependency_depth = check_int(
            dependency_depth, "dependency_depth", l_bound=1
        )
        self._random_circuit_mode = self._parse_random_circuit_mode(random_circuit_mode)
        self._commutation_rulebook = self._parse_rulebook(rulebook)
        self._rewarder = self._parse_rewarder(rewarder)

        self._gate_encoder = machine_properties.encode()
        self._random_circuit_generator = RandomCircuitGenerator(
            machine_properties.n_qubits, max_gates, rng=self.rng
        )

        self._state = {
            "max_gates": max_gates,
            "n_qubits": machine_properties.n_qubits,
            "gate_cycle_length": machine_properties.gates,
            "same_start": machine_properties.same_start,
            "not_in_same_cycle": machine_properties.not_in_same_cycle,
        }

        self.reset()

        legal_actions_space = qgym.spaces.MultiBinary(max_gates, rng=self.rng)
        gate_names_space = qgym.spaces.MultiDiscrete(
            np.full(max_gates, machine_properties.n_gates + 1), rng=self.rng
        )
        acts_on_space = qgym.spaces.MultiDiscrete(
            np.full(2 * max_gates, machine_properties.n_qubits + 1), rng=self.rng
        )
        dependencies_space = qgym.spaces.MultiDiscrete(
            np.full(dependency_depth * max_gates, max_gates), rng=self.rng
        )

        self.observation_space = qgym.spaces.Dict(
            rng=self.rng,
            legal_actions=legal_actions_space,
            gate_names=gate_names_space,
            acts_on=acts_on_space,
            dependencies=dependencies_space,
        )

        self.action_space = qgym.spaces.MultiDiscrete([max_gates, 2], rng=self.rng)

        self._visualiser = SchedulingVisualiser(
            gate_encoder=self._gate_encoder,
            gate_cycle_length=machine_properties.gates,
            n_qubits=machine_properties.n_qubits,
        )

    def _obtain_observation(self) -> Dict[str, NDArray[np.int_]]:
        """:return: Observation based on the current state."""
        return {
            "gate_names": self._state["gate_names"],
            "acts_on": self._state["acts_on"].flatten(),
            "dependencies": self._state["dependencies"].flatten(),
            "legal_actions": self._state["legal_actions"],
        }

    def _update_state(self, action: NDArray[np.int_]) -> None:
        """Update the state of this environment using the given action.

        :param action: First entry determines a gate to schedule, the second entry
            increases the cycle if nonzero.
        """
        # Increase the step number
        self._state["steps_done"] += 1

        # Increase the cycle if the action is given
        if action[1]:
            self._increment_cycle()
            return

        # Schedule the gate if it is allowed
        gate_to_schedule = action[0]
        if self._state["legal_actions"][gate_to_schedule]:
            self._schedule_gate(gate_to_schedule)

    def _increment_cycle(self) -> None:
        """Increment the cycle and update the state accordingly."""
        self._state["cycle"] += 1

        # Reduce the amount of cycles each qubit is busy
        self._state["busy"][self._state["busy"] > 0] -= 1

        # Exclude gates that should start at the same time
        while len(self._exclude_in_next_cycle) != 0:
            gate_to_exclude = self._exclude_in_next_cycle.pop()
            self._exclude_gate(gate_to_exclude)

        # Decrease the amount of cycles to exclude a gate and skip gates where the
        # cycle becomes 0 (as it no longer should be excluded)
        gate_names = list(self._excluded_gates.keys())
        for gate_name in gate_names:
            if self._excluded_gates[gate_name] > 1:
                self._excluded_gates[gate_name] -= 1
            else:
                self._excluded_gates.pop(gate_name)

        self._update_legal_actions()

    def _schedule_gate(self, gate_idx: int) -> None:
        """Schedule a gate in the current cycle and update the state accordingly.

        :param gate_idx: Index of the gate to schedule.
        """
        gate = self._state["encoded_circuit"][gate_idx]

        # add the gate to the schedule
        self._state["schedule"][gate_idx] = self._state["cycle"]

        self._state["busy"][gate.q1] = self._state["gate_cycle_length"][gate.name]
        self._state["busy"][gate.q2] = self._state["gate_cycle_length"][gate.name]

        if gate.name in self._state["not_in_same_cycle"]:
            for gate_to_exclude in self._state["not_in_same_cycle"][gate.name]:
                self._exclude_gate(gate_to_exclude)

        if gate.name in self._state["same_start"]:
            self._exclude_in_next_cycle.add(gate.name)

        # Update "dependencies" observation
        self._blocking_matrix[:gate_idx, gate_idx] = False
        self._state["dependencies"] = self._get_dependencies()

        self._update_legal_actions()

    def _is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        return bool((self._state["schedule"] != -1).all())

    def _obtain_info(self) -> Dict[str, Any]:
        """:return: Optional debugging info for the current state."""
        return {
            "Steps done": self._state["steps_done"],
            "Cycle": self._state["cycle"],
            "Schedule": self._state["schedule"],
        }

    def reset(
        self,
        *,
        circuit: Optional[List[Gate]] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
    ) -> Union[
        Tuple[NDArray[np.int_], NDArray[np.int_]],
        Tuple[Tuple[NDArray[np.int_], NDArray[np.int_]], Dict[Any, Any]],
    ]:
        """Reset the state, action space and step number and load a new (random) initial
        state. To be used after an episode is finished.

        :param circuit: Optional list of a circuit for the next episode, each entry in
            the list should be a ``Gate``. When a circuit is give, no random circuit
            will be generated.
        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call, i.e., before any learning is done.
        :param return_info: Whether to receive debugging info.
        :return: Initial observation and optionally debugging info.
        """
        # Reset counters
        self._state["steps_done"] = 0
        self._state["cycle"] = 0

        # Amount of cycles that a qubit is still busy (zero if available)
        self._state["busy"] = np.zeros(self._state["n_qubits"], dtype=int)

        # At the start no gates should be excluded
        self._excluded_gates = {}
        self._exclude_in_next_cycle = set()

        # Generate a circuit if None is given
        if circuit is None:
            circuit = self._random_circuit_generator.generate_circuit(
                mode=self._random_circuit_mode
            )
        self._blocking_matrix = self._commutation_rulebook.make_blocking_matrix(circuit)
        self._state["dependencies"] = self._get_dependencies()

        encoded_circuit = self._gate_encoder.encode_gates(circuit)
        self._state["encoded_circuit"] = encoded_circuit

        self._state["schedule"] = np.full(len(circuit), -1, dtype=int)

        self._update_episode_constant_observations()
        self._update_legal_actions()

        # call super method for dealing with the general stuff
        return super().reset(seed=seed, return_info=return_info)

    def _get_dependencies(self) -> NDArray[np.int_]:
        """Compute the dependencies array of the current state.

        :return: array of shape (dependency_depth, max_gates) with the dependencies
            for each gate.
        """
        dependencies = np.zeros(
            (self._dependency_depth, self._state["max_gates"]), dtype=int
        )
        for gate_idx, blocking_row in enumerate(self._blocking_matrix):
            blocking_gates = blocking_row[gate_idx:].nonzero()[0]
            for depth in range(min(self._dependency_depth, blocking_gates.shape[0])):
                dependencies[depth, gate_idx] = blocking_gates[depth]

        return dependencies

    def _update_episode_constant_observations(self) -> None:
        """Update episode constant observations `gate_names` and `acts_on` based on
        the circuit of the current episode.
        """
        circuit = self._state["encoded_circuit"]

        gate_names = np.zeros(self._state["max_gates"], dtype=int)
        acts_on = np.zeros((2, self._state["max_gates"]), dtype=int)

        for gate_idx, gate in enumerate(circuit):
            gate_names[gate_idx] = gate.name
            acts_on[0, gate_idx] = gate.q1
            acts_on[1, gate_idx] = gate.q2

        self._state["gate_names"] = gate_names
        self._state["acts_on"] = acts_on

    def _exclude_gate(self, gate_name: int) -> None:
        """Exclude a gate from the 'legal_actions' for 'gate_cycle_length' cycles.

        :param gate_name: integer encoding of the name of the gate.
        """
        gate_cycle_length = self._state["gate_cycle_length"][gate_name]
        self._excluded_gates[gate_name] = gate_cycle_length

    def _update_legal_actions(self) -> None:
        """Check which actions are legal based on the scheduled qubits. An action is
        legal if the gate could be scheduled based on the machine properties and
        commutation rules.
        """
        legal_actions = np.zeros(self._state["max_gates"], dtype=bool)
        for gate_idx, (gate_name, qubit1, qubit2) in enumerate(
            self._state["encoded_circuit"]
        ):
            # Set all gates, which have not been scheduled to True
            if self._state["schedule"][gate_idx] == -1:
                legal_actions[gate_idx] = True

            # Check if there is a non-scheduled dependent gate

            dependent_gates = self._state["dependencies"][:, gate_idx]
            if np.count_nonzero(dependent_gates) > 0:
                legal_actions[gate_idx] = False
                continue

            # Check if the qubits are busy
            if self._state["busy"][qubit1] > 0 or self._state["busy"][qubit2] > 0:
                legal_actions[gate_idx] = False
                continue

            # Check if gates should be excluded
            if gate_name in self._excluded_gates:
                legal_actions[gate_idx] = False
                continue

        self._state["legal_actions"] = legal_actions

    def get_circuit(self, mode: str = "human") -> List[Gate]:
        """Return the quantum circuit of this episode.

        :param mode: Choose from be 'human' or 'encoded'. Defaults to 'human'.
        :raise ValueError: If an unsupported mode is provided.
        :return: Human or encoded quantum circuit.
        """
        if not isinstance(mode, str):
            raise TypeError(f"mode must be of type str, but was {type(mode)}")

        if mode.lower() == "encoded":
            return deepcopy(self._state["encoded_circuit"])

        elif mode.lower() == "human":
            return self._gate_encoder.decode_gates(self._state["encoded_circuit"])

        else:
            raise ValueError(f"mode must be 'human' or 'encoded', but was {mode}")

    def render(self, mode: str = "human") -> Any:
        """Render the current state using pygame.

        :param mode: The mode to render with (supported modes are found in
            `self.metadata`.).
        :raise ValueError: If an unsupported mode is provided.
        :return: Result of rendering.
        """
        if mode not in self.metadata["render.modes"]:
            raise ValueError("The given render mode is not supported.")

        return self._visualiser.render(self._state, mode)

    def close(self) -> None:
        """Close the screen used for rendering."""
        if hasattr(self, "_visualiser"):
            self._visualiser.close()

    @staticmethod
    def _parse_machine_properties(
        machine_properties: Union[Mapping[str, Any], str, MachineProperties]
    ) -> MachineProperties:
        """
        Parse the machine_properties given by the user and return a
        ``MachineProperties`` object.

        :param machine_properties: A ``MachineProperties`` object, a ``Mapping`` of
            machine or a string with a filename for a file containing the machine
            properties.
        :raise TypeError: If the given type is not supported.
        :return: ``MachineProperties`` object with the given machine properties.
        """
        if isinstance(machine_properties, str):
            return MachineProperties.from_file(machine_properties)
        elif isinstance(machine_properties, MachineProperties):
            return deepcopy(machine_properties)
        elif isinstance(machine_properties, Mapping):
            return MachineProperties.from_mapping(machine_properties)
        else:
            msg = f"{type(machine_properties)} is not a supported type for "
            msg += "'machine_properties'"
            raise TypeError(msg)

    @staticmethod
    def _parse_random_circuit_mode(random_circuit_mode: str) -> str:
        """Parse the `random_circuit_mode` given by the user.

        :param random_circuit_mode: Mode for the random circuit generator. The mode
            can be 'default' or 'workshop'.
        :raise ValueError: If the given mode is not supported.
        :return: Valid random circuit mode.
        """
        random_circuit_mode = check_string(
            random_circuit_mode, "random_circuit_mode", lower=True
        )
        if random_circuit_mode not in ["default", "workshop"]:
            msg = f"'{random_circuit_mode}' is not a supported random circuit mode"
            raise ValueError(msg)
        return random_circuit_mode

    @staticmethod
    def _parse_rulebook(
        rulebook: Union[CommutationRulebook, None]
    ) -> CommutationRulebook:
        """Parse the `rulebook` given by the user.

        :param rulebook: Rulebook describing the commutation rules. If ``None`` is
            given, a default ``CommutationRulebook`` will be used. (See
            ``CommutationRulebook`` for more info on the default rules.)
        :return: ``CommutationRulebook``.
        """
        if rulebook is None:
            return CommutationRulebook()
        check_instance(rulebook, "rulebook", CommutationRulebook)
        return deepcopy(rulebook)

    @staticmethod
    def _parse_rewarder(rewarder: Union[Rewarder, None]) -> Rewarder:
        """Parse the rewarder given by the user.

        :param rewarder: Rewarder to use for the environment. If ``None``, then a
            ``BasicRewarder`` is used.
        :return: Rewarder.
        """
        if rewarder is None:
            return BasicRewarder()
        check_instance(rewarder, "rewarder", Rewarder)
        return deepcopy(rewarder)
