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

    * `machine_properties`: MachineProperties object containing machine properties
      and limitations.
    * `utils`: ``SchedulingUtils`` dataclass with a random circuit generator,
      commutation rulebook and a gate encoder.
    * `gates`: Dictionary with gate names as keys and ``GateInfo`` dataclasses as
      values.
    * `steps_done`: Number of steps done since the last reset.
    * `cycle`: Current 'machine' cycle.
    * `busy`: Used internally for the hardware limitations.
    * `circuit_info`: ``CircuitInfo`` dataclass containing the encoded circuit and
      attributes used to update the state.

Observation space:
    Each element in the observation space is a dictionary with 4 entries:

    * `gate_names`: Gate names of the encoded circuit.
    * `acts_on`: q1 and q2 of each gate.
    * `dependencies`: Shows the first $n$ gates that must be scheduled before this
      gate.
    * `legal_actions`: List of legal actions. If the value at index $i$ determines if
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
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.custom_types import Gate
from qgym.envs.scheduling.machine_properties import MachineProperties
from qgym.envs.scheduling.rulebook import CommutationRulebook
from qgym.envs.scheduling.scheduling_rewarders import BasicRewarder
from qgym.envs.scheduling.scheduling_state import SchedulingState
from qgym.envs.scheduling.scheduling_visualiser import SchedulingVisualiser
from qgym.templates import Environment, Rewarder
from qgym.utils.input_parsing import parse_rewarder
from qgym.utils.input_validation import check_instance, check_int, check_string


class Scheduling(
    Environment[Dict[str, Union[NDArray[np.int_], NDArray[np.int8]]], NDArray[np.int_]]
):
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
        dependency_depth = check_int(dependency_depth, "dependency_depth", l_bound=1)
        random_circuit_mode = self._parse_random_circuit_mode(random_circuit_mode)
        rulebook = self._parse_rulebook(rulebook)
        self._rewarder = parse_rewarder(rewarder, BasicRewarder)

        self._state = SchedulingState(
            machine_properties=machine_properties,
            max_gates=max_gates,
            dependency_depth=dependency_depth,
            random_circuit_mode=random_circuit_mode,
            rulebook=rulebook,
        )
        self.observation_space = self._state.create_observation_space()
        self.action_space = qgym.spaces.MultiDiscrete([max_gates, 2], rng=self.rng)

        self._visualiser = SchedulingVisualiser(self._state)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        circuit: Optional[List[Gate]] = None,
        **_kwargs: Any,
    ) -> Union[
        Dict[str, Union[NDArray[np.int_], NDArray[np.int8]]],
        Tuple[Dict[str, Union[NDArray[np.int_], NDArray[np.int8]]], Dict[Any, Any]],
    ]:
        """Reset the state, action space and step number.and load a new (random) initial
        state. To be used after an episode is finished.

        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call, i.e., before any learning is done.
        :param return_info: Whether to receive debugging info.
        :return: Initial observation and optionally debugging info.
        :param circuit: Optional list of a circuit for the next episode, each entry in
            the list should be a ``Gate``. When a circuit is give, no random circuit
            will be generated.
        :param _kwargs: Additional options to configure the reset.
        :return: Initial observation and optionally also debugging info.
        """
        # call super method for dealing with the general stuff
        return super().reset(seed=seed, return_info=return_info, circuit=circuit)

    def get_circuit(self, mode: str = "human") -> List[Gate]:
        """Return the quantum circuit of this episode.

        :param mode: Choose from be 'human' or 'encoded'. Defaults to 'human'.
        :raise ValueError: If an unsupported mode is provided.
        :return: Human or encoded quantum circuit.
        """
        mode = check_string(mode, "mode", lower=True)
        state = cast(SchedulingState, self._state)
        encoded_circuit = state.circuit_info.encoded
        if mode == "encoded":
            return deepcopy(encoded_circuit)

        if mode == "human":
            gate_encoder = state.utils.gate_encoder
            return gate_encoder.decode_gates(encoded_circuit)

        raise ValueError(f"mode must be 'human' or 'encoded', but was {mode}")

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
        if isinstance(machine_properties, MachineProperties):
            return deepcopy(machine_properties)
        if isinstance(machine_properties, Mapping):
            return MachineProperties.from_mapping(machine_properties)

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
