"""This module contains the :class:`MachineProperties` class, which helps with
conveniently setting up machine properties for the scheduling environment.

Usage:
    The code below will create :class:`MachineProperties` which will have the following
    properties:

    * The machine has two qubits.
    * The machine supports the X, Y, C-NOT and Measure gates.
    * Multiple Measure gates should start in the same cycle, or wait till the previous
      one is done.
    * The X and Y gates cannot be performed in the same cycle.

    .. code-block:: python

        from qgym.envs.scheduling import MachineProperties

        hardware_spec = MachineProperties(n_qubits=2)
        hardware_spec.add_gates({"x": 2, "y": 2, "cnot": 4, "measure": 10})
        hardware_spec.add_same_start(["measure"])
        hardware_spec.add_not_in_same_cycle([("x", "y")])


"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Iterable, Mapping

from qgym.utils import GateEncoder
from qgym.utils.input_validation import check_instance, check_int, check_string


class MachineProperties:
    """:class:`MachineProperties` is a class to conveniently setup machine properties
    for the :class:`~qgym.envs.Scheduling` environment.
    """

    def __init__(self, n_qubits: int) -> None:
        """
        Init of the MachineProperties class.

        Args:
            n_qubits: Number of qubits of the machine.
        """
        self._n_qubits = check_int(n_qubits, "n_qubits", l_bound=1)
        self._gates: dict[Any, int] = {}
        self._same_start: set[Any] = set()
        self._not_in_same_cycle: dict[Any, set[Any]] = {}

    @classmethod
    def from_mapping(cls, machine_properties: Mapping[str, Any]) -> MachineProperties:
        """Initialize the :class:`MachineProperties` class from a ``Mapping`` containing
        valid machines properties.

        Args:
            machine_properties: ``Mapping`` containing valid machine properties.

        Returns:
            Initialized :class:`MachineProperties` object with the properties described
            in the `machine_properties` ``Mapping``.
        """
        checked_mp = cls._check_machine_properties_mapping(machine_properties)

        _machine_properties = cls(checked_mp["n_qubits"])
        _machine_properties.add_gates(checked_mp["gates"])
        _machine_properties.add_same_start(checked_mp["same_start"])
        _machine_properties.add_not_in_same_cycle(checked_mp["not_in_same_cycle"])
        return _machine_properties

    @classmethod
    def from_file(cls, filename: str) -> MachineProperties:
        """Load MachineProperties from a JSON file. Not implemented."""
        raise NotImplementedError(
            "Loading machine properties from files is not yet implemented."
        )

    def add_gates(self, gates: Mapping[str, int]) -> MachineProperties:
        """Add gates to the machine properties that should be supported.

        Args:
            gates: ``Mapping`` of gates that the machine can perform as keys, and the
                number of machine cycles (time) as values.

        Returns:
            The :class:`MachineProperties` with the added gates.
        """
        check_instance(gates, "gates", Mapping)

        for gate_name, n_cycles in gates.items():
            gate_name = check_string(gate_name, "gate name", lower=True)
            n_cycles = check_int(n_cycles, "n cycles", l_bound=1)
            if gate_name in self._gates:
                msg = f"Gate '{gate_name}' was already given. Overwriting it with the "
                msg += "new value."
                warnings.warn(msg)
            self._gates[gate_name] = n_cycles
            self._not_in_same_cycle[gate_name] = set()
        return self

    def add_same_start(self, gates: Iterable[str]) -> MachineProperties:
        """Add gates that should start in the same cycle, or wait till the previous gate
        is done.

        Args:
            gates: ``Iterable`` of gate names that should start in the same cycle.

        Returns:
            The :class:`MachineProperties` with the same start gates.
        """
        check_instance(gates, "gates", Iterable)

        for gate_name in gates:
            gate_name = check_string(gate_name, "gate name", lower=True)
            if gate_name not in self.gates:
                raise ValueError(f"unknown gate '{gate_name}'")
            self._same_start.add(gate_name)
        return self

    def add_not_in_same_cycle(
        self, gates: Iterable[tuple[str, str]]
    ) -> MachineProperties:
        """Add gates that should not start in the same cycle.

        Args:
            gates: ``Iterable`` of tuples of gate names that should not start in the
                same cycle.

        Returns:
            The :class:`MachineProperties` with an updated
            :class:`~MachineProperties.not_in_same_cycle` property.
            The :class:`~MachineProperties.not_in_same_cycle` property is updated
            according to the input `gates`.
        """
        check_instance(gates, "gates", Iterable)
        for gate1, gate2 in gates:
            # Check if the gates are strings and known.
            gate1 = check_string(gate1, "gate", lower=True)
            gate2 = check_string(gate2, "gate", lower=True)
            if gate1 not in self.gates:
                raise ValueError(f"unknown gate '{gate1}'")
            if gate2 not in self.gates:
                raise ValueError(f"unknown gate '{gate2}'")

            self._not_in_same_cycle[gate1].add(gate2)
            self._not_in_same_cycle[gate2].add(gate1)

        return self

    def encode(self) -> GateEncoder:
        """Encode the gates in the machine properties to integer values.

        Returns:
            The :class:`~qgym.utils.GateEncoder` used to encode the gates. This
            :class:`~qgym.utils.GateEncoder` can be used to decode the gates or encode
            quantum circuits containing the same gate names as in this
            :class:`MachineProperties` object.
        """
        if (
            any(not isinstance(gate, str) for gate in self._gates)
            or any(not isinstance(gate, str) for gate in self._same_start)
            or any(not isinstance(gate, str) for gate in self._not_in_same_cycle)
        ):
            msg = "Gate names of machine properties are not of type str, perhaps they "
            msg = "are already encoded?"
            raise ValueError(msg)

        gate_encoder = GateEncoder().learn_gates(self._gates)

        self._gates = gate_encoder.encode_gates(self._gates)
        self._same_start = gate_encoder.encode_gates(self._same_start)
        self._not_in_same_cycle = gate_encoder.encode_gates(self._not_in_same_cycle)
        return gate_encoder

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits of the machine."""
        return self._n_qubits

    @property
    def gates(self) -> dict[str, int] | dict[int, int]:
        """Return a``Dict`` with the gate names the machine can perform as keys, and the
        number of machine cycles (time) as values.
        """
        return self._gates

    @property
    def n_gates(self) -> int:
        """Return the number of supported gates."""
        return len(self._gates)

    @property
    def same_start(self) -> set[str] | set[int]:
        """Set of gate names that should start in the same cycle, or wait till the
        previous gate is done.
        """
        return self._same_start

    @property
    def not_in_same_cycle(self) -> dict[str, set[str]] | dict[int, set[int]]:
        """Gates that can not start in the same cycle."""
        return self._not_in_same_cycle

    @staticmethod
    def _check_machine_properties_mapping(
        machine_properties: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Check if the given machine properties ``Mapping`` is a valid descriptions of
        the machine properties and returns a ``Dict`` to easily initialize a
        ``MachineProperties`` object.

        Args:
            machine_properties: ``Mapping`` containing the machine properties.

        Raises:
            TypeError: If ``machine_properties`` is not a ``Mapping``.
            ValueError: If there are missing keys in the ``Mapping``.

        Returns:
            ``Dict`` to easily initialize a ``MachineProperties`` object.
        """
        check_instance(machine_properties, "machine_properties", Mapping)
        keys = ["n_qubits", "gates", "machine_restrictions"]
        if not all(key in machine_properties for key in keys):
            msg = "'machine_properties' must have the keys 'n_qubits', 'gates' and "
            msg += "'machine_restrictions'"
            raise ValueError(msg)

        checked_mp = {}
        checked_mp["n_qubits"] = deepcopy(machine_properties["n_qubits"])
        checked_mp["gates"] = deepcopy(machine_properties["gates"])

        machine_restrictions = machine_properties["machine_restrictions"]
        check_instance(machine_restrictions, "machine_restrictions", Mapping)
        keys = ["same_start", "not_in_same_cycle"]
        if not all(key in machine_restrictions for key in keys):
            msg = "'machine_restrictions' must have the keys 'same_start' and "
            msg += "'not_in_same_cycle'"
            raise ValueError(msg)

        checked_mp["same_start"] = deepcopy(machine_restrictions["same_start"])
        checked_mp["not_in_same_cycle"] = []

        for gate1, gate_lst in machine_restrictions["not_in_same_cycle"].items():
            for gate_other in gate_lst:
                checked_mp["not_in_same_cycle"].append((gate1, gate_other))

        return checked_mp

    def __str__(self) -> str:
        """Make a string representation of the machine properties."""
        text = f"{self.__class__.__name__}:\n"
        text += f"n_qubits: {self._n_qubits}\n"
        text += f"gates: {self._gates}\n"
        text += f"same_start: {self._same_start}\n"
        text += f"not_in_same_cycle: {self._not_in_same_cycle}"
        return text

    def __repr__(self) -> str:
        """Make a string representation without endline characters."""
        text = f"{self.__class__.__name__}("
        text += f"n_qubits={self._n_qubits}, "
        text += f"gates={self._gates}, "
        text += f"same_start={self._same_start}, "
        text += f"not_in_same_cycle={self._not_in_same_cycle})"
        return text
