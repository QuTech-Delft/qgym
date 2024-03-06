r"""This module contains an environment for training an RL agent on the quantum
operation scheduling problem of OpenQL. The quantum operations scheduling problem is
aimed at finding the shortest possible schedules of operations defined by a **quantum
circuit**, whilst taking **hardware constraints** and **commutation rules** into
account.

Circuits:
    The quantum operations scheduling problem consist of scheduling the operations
    described by a quantum circuit. For the :class:`Scheduling` environment in this
    module, a quantum circuit is defined as a list of gates, where a gate is a
    ``namedtuple`` with a 'name', 'q1' and 'q2'. For example, a c-not with control
    qubit 2 and target qubit 5 would be ``Gate("cnot", 2, 5)`` and an X gate acting on
    qubit 4 would be ``Gate("x", 4, 4)``. In this representation, the circuit below can
    be created by:

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
    hardware. To be more concrete, the :class:`Scheduling` environment can take two more
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
    * `utils`: :class:`~qgym.envs.scheduling.scheduling_dataclasses.SchedulingUtils`
      dataclass with a circuit generator, commutation rulebook and a gate encoder.
    * `gates`: Dictionary with gate names as keys and
      :class:`~qgym.envs.scheduling.scheduling_dataclasses.GateInfo` dataclasses as
      values.
    * `steps_done`: Number of steps done since the last reset.
    * `cycle`: Current 'machine' cycle.
    * `busy`: Used internally for the hardware limitations.
    * `circuit_info`: :class:`~qgym.envs.scheduling.scheduling_dataclasses.CircuitInfo`
      dataclass containing the encoded circuit and attributes used to update the state.

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

    The code block below shows how a :class:`Scheduling` environment for such a machine,
    were the environment only allows for circuit with a maximum length of 5:

    .. code-block:: python

        from qgym.envs.scheduling import Scheduling, MachineProperties
        from qgym.generators import WorkshopCircuitGenerator

        # Set up the hardware specifications
        hardware_spec = MachineProperties(n_qubits=2)
        hardware_spec.add_gates({"x": 2, "y": 2, "cnot": 4, "measure": 10})
        hardware_spec.add_same_start(["measure"])
        hardware_spec.add_not_in_same_cycle([("x", "y")])

        # Set up the circuit generator
        circuit_generator = WorkshopCircuitGenerator(2, 10)

        # Initialize the environment
        env = Scheduling(
            hardware_spec,
            max_gates=5,
            circuit_generator=circuit_generator
        )


Example 2:
    This example shows how to add a specified commutation rulebook, which can contain
    costum commutation rules. For more information on implementing custom cummutation
    rules, see the documentation of :class:`~qgym.envs.scheduling.CommutationRulebook`.
    This example uses the same machine properties as example 1. Custom commutation rules
    can be added as shown in the code block below:

    .. code-block:: python

        from qgym.envs.scheduling import Scheduling, MachineProperties
        from qgym.envs.scheduling.rulebook import CommutationRulebook
        from qgym.generators import WorkshopCircuitGenerator

        # Set up the hardware specifications
        hardware_spec = MachineProperties(n_qubits=2)
        hardware_spec.add_gates({"x": 2, "y": 2, "cnot": 4, "measure": 10})
        hardware_spec.add_same_start(["measure"])
        hardware_spec.add_not_in_same_cycle([("x", "y")])

        # Setup the rulebook
        rulebook = CommutationRulebook()

        # Set up the circuit generator
        circuit_generator = WorkshopCircuitGenerator(2, 10)

        # Initialize the environment
        env_com = Scheduling(
                            hardware_spec,
                            max_gates=5,
                            rulebook=rulebook,
                            circuit_generator=circuit_generator,
                        )


"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Union, cast

import numpy as np
from numpy.typing import NDArray

import qgym.spaces
from qgym.custom_types import Gate
from qgym.envs.scheduling.machine_properties import MachineProperties
from qgym.envs.scheduling.rulebook import CommutationRulebook
from qgym.envs.scheduling.scheduling_rewarders import BasicRewarder
from qgym.envs.scheduling.scheduling_state import SchedulingState
from qgym.envs.scheduling.scheduling_visualiser import SchedulingVisualiser
from qgym.generators.circuit import BasicCircuitGenerator, CircuitGenerator
from qgym.templates import Environment, Rewarder
from qgym.utils.input_parsing import parse_rewarder, parse_visualiser
from qgym.utils.input_validation import check_instance, check_int, check_string


class Scheduling(
    Environment[Dict[str, Union[NDArray[np.int_], NDArray[np.int8]]], NDArray[np.int_]]
):
    """RL environment for the scheduling problem."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        machine_properties: Mapping[str, Any] | str | MachineProperties,
        *,
        max_gates: int = 200,
        dependency_depth: int = 1,
        circuit_generator: CircuitGenerator | None = None,
        rulebook: CommutationRulebook | None = None,
        rewarder: Rewarder | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the action space, observation space, and initial states for the
        scheduling environment.

        Args:
            machine_properties: A :class:`~qgym.envs.scheduling.MachineProperties`
                object, a ``Mapping`` containing machine properties or a string with a
                filename for a file containing the machine properties.
            max_gates: Maximum number of gates allowed in a circuit. Defaults to 200.
            dependency_depth: Number of dependencies given in the observation.
                Determines the shape of the `dependencies` observation, which has the
                shape (dependency_depth, max_gates). Defaults to 1.
            circuit_generator: Generator class for generating circuits for training.
            rulebook: :class:`~qgym.envs.scheduling.CommutationRulebook` describing the
                commutation rules. If ``None`` (default) is given, a default
                :class:`~qgym.envs.scheduling.CommutationRulebook` will be used. (See
                :class:`~qgym.envs.scheduling.CommutationRulebook` for more info on the
                default rules.)
            rewarder: Rewarder to use for the environment. If ``None`` (default), then a
                default :class:`~qgym.envs.scheduling.BasicRewarder` is used.
            render_mode: If ``"human"`` open a ``pygame`` screen visualizing the step.
                If ``"rgb_array"``, return an RGB array encoding of the rendered frame
                on each render call.
        """
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "random_circuit.mode": ["default", "workshop"],
        }

        machine_properties = self._parse_machine_properties(machine_properties)
        max_gates = check_int(max_gates, "max_gates", l_bound=1)

        if circuit_generator is None:
            circuit_generator = BasicCircuitGenerator(
                machine_properties.n_qubits, max_gates, seed=self.rng
            )
        else:
            check_instance(circuit_generator, "circuit_generator", CircuitGenerator)
            if circuit_generator.finite:
                raise ValueError(
                    "'circuit_generator' should not be an infinite iterator"
                )
            circuit_generator = deepcopy(circuit_generator)

        dependency_depth = check_int(dependency_depth, "dependency_depth", l_bound=1)
        rulebook = self._parse_rulebook(rulebook)
        self._rewarder = parse_rewarder(rewarder, BasicRewarder)

        self._state = SchedulingState(
            machine_properties=machine_properties,
            max_gates=max_gates,
            dependency_depth=dependency_depth,
            circuit_generator=circuit_generator,
            rulebook=rulebook,
        )
        self.observation_space = self._state.create_observation_space()
        self.action_space = qgym.spaces.MultiDiscrete([max_gates, 2], rng=self.rng)

        self._visualiser = parse_visualiser(
            render_mode, SchedulingVisualiser, [self._state]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.int_] | NDArray[np.int8]], dict[str, Any]]:
        r"""Reset the state, action space and load a new (random) initial state.

        To be used after an episode is finished.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call, i.e., before any learning is done.
            return_info: Whether to receive debugging info.
            options: Mapping with keyword arguments with additional options for the
                reset. Keywords can be found in the description of
                :class:`~qgym.envs.scheduling.SchedulingState`.\
                :class:`~qgym.envs.scheduling.SchedulingState.reset`.
            _kwargs: Additional options to configure the reset.

        Returns:
            Initial observation and debugging info.
        """
        return super().reset(seed=seed, options=options)

    def get_circuit(self, mode: str = "human") -> list[Gate]:
        """Return the quantum circuit of this episode.

        Args:
            mode: Choose from be ``"human"`` or ``"encoded"``. Defaults to ``"human"``.

        Raises:
            ValueError: If an unsupported mode is provided.

        Returns:
            Human or encoded quantum circuit.
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
        machine_properties: Mapping[str, Any] | str | MachineProperties
    ) -> MachineProperties:
        """
        Parse the machine_properties given by the user and return a
        ``MachineProperties`` object.

        Args:
            machine_properties: A ``MachineProperties`` object, a ``Mapping`` of machine
            or a string with a filename for a file containing the machine properties.

        Raises:
            TypeError: If the given type is not supported.

        Returns:
            ``MachineProperties`` object with the given machine properties.
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
    def _parse_rulebook(rulebook: CommutationRulebook | None) -> CommutationRulebook:
        """Parse the `rulebook` given by the user.

        Args:
            rulebook: Rulebook describing the commutation rules. If ``None`` is given, a
                default ``CommutationRulebook`` will be used. (See
                ``CommutationRulebook`` for more info on the default rules.)

        Returns:
            ``CommutationRulebook``.
        """
        if rulebook is None:
            return CommutationRulebook()
        check_instance(rulebook, "rulebook", CommutationRulebook)
        return deepcopy(rulebook)
