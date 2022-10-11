"""
This module contains the MachineProperties class, which helps with conveniently setting
up machine properties for the scheduling environment.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

from qgym.utils import GateEncoder
from qgym.utils.input_validation import check_instance, check_int, check_string


class MachineProperties:
    """
    MachineProperties is a class to conveniently setup machine properties for the
    scheduling environment.
    """

    def __init__(self, n_qubits: int) -> None:
        """
        Init of the MachineProperties class.

        :param n_qubits: Number of qubits of the machine.
        """

        self._n_qubits = check_int(n_qubits, "n_qubits", l_bound=1)
        self._gates = {}
        self._same_start = set()
        self._not_in_same_cycle = {}

    @classmethod
    def from_mapping(cls, machine_properties: Mapping[str, Any]) -> MachineProperties:
        """
        Initialize the MachineProperties class from Mapping containing the machines
        properties.

        :param machine_properties: Mapping containing the machine properties.
        :return: Initialized MachineProperties object with the properties described in
            the 'machine_properties' Mapping.
        """
        checked_mp = cls._check_machine_properties_mapping(machine_properties)

        mp = cls(checked_mp["n_qubits"])
        mp.add_gates(checked_mp["gates"])
        mp.add_same_start(checked_mp["same_start"])
        mp.add_not_in_same_cycle(checked_mp["not_in_same_cycle"])
        return mp

    @classmethod
    def from_file(cls, filename: str) -> MachineProperties:
        raise NotImplementedError(
            "Loading machine properties from files is not yet implemented."
        )

    def add_gates(self, gates: Mapping[str, int]) -> MachineProperties:
        """
        Add possible gates to the machine.

        :param gates: Mapping of gates that the machine can perform as keys and the
            number of machine cycles (time) as values.
        :return: The MachineProperties with the added gates.
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
        return self

    def add_same_start(self, gates: Iterable[str]) -> MachineProperties:
        """
        Add gates that should start in the same cycle, or wait till the previous gate
        is done.

        :param gates: Iterable of gates that should start in the same cycle.
        :return: The MachineProperties with the same start gates.
        """
        check_instance(gates, "gates", Iterable)

        for gate_name in gates:
            gate_name = check_string(gate_name, "gate name", lower=True)
            if gate_name not in self.gates:
                raise ValueError(f"unknown gate '{gate_name}'")
            self._same_start.add(gate_name)
        return self

    def add_not_in_same_cycle(
        self, gates: Iterable[Tuple[str, str]]
    ) -> MachineProperties:
        """
        Add gates that should not start in the same cycle.

        :param gates: Iterable of tuples of gates that should not start in the same
            cycle.
        :return: The MachineProperties with the not in same start gates.
        """
        check_instance(gates, "gates", Iterable)
        for (gate1, gate2) in gates:

            # Check if the gates are strings and known.
            gate1 = check_string(gate1, "gate", lower=True)
            gate2 = check_string(gate2, "gate", lower=True)
            if gate1 not in self.gates:
                raise ValueError(f"unknown gate '{gate1}'")
            if gate2 not in self.gates:
                raise ValueError(f"unknown gate '{gate2}'")

            if gate1 in self.not_in_same_cycle:
                if gate2 not in self.not_in_same_cycle[gate1]:
                    self.not_in_same_cycle[gate1].append(gate2)
            else:
                self.not_in_same_cycle[gate1] = [gate2]

            if gate2 in self.not_in_same_cycle:
                if gate1 not in self.not_in_same_cycle[gate2]:
                    self.not_in_same_cycle[gate2].append(gate1)
            else:
                self.not_in_same_cycle[gate2] = [gate1]

    def encode(self) -> GateEncoder:
        """
        Encode the gates in the machine properties to integer values.

        :return: The GateEncoder used to encode the gates. This GateEncoder can be used
            to decode the gates or encode quantum circuits containing  the same gate
            names as in this MachineProperties object.
        """
        gate_encoder = GateEncoder().learn_gates(self.gates)
        self._gates = gate_encoder.encode_gates(self.gates)
        self._same_start = gate_encoder.encode_gates(self.same_start)
        self._not_in_same_cycle = gate_encoder.encode_gates(self.not_in_same_cycle)
        return gate_encoder

    @property
    def n_qubits(self) -> int:
        """
        Number of qubits of the machine.
        """

        return self._n_qubits

    @property
    def gates(self) -> Dict[str, int]:
        """
        Dictionary with the gates the machine can perform as keys and the number of
        machine cycles (time) as values.
        """

        return self._gates

    @property
    def n_gates(self) -> Dict[str, int]:
        """
        Number of gates
        """
        return len(self._gates)

    @property
    def same_start(self) -> Set:
        """
        Set of gates that should start in the same cycle, or wait till the previous gate
        is done.
        """
        return self._same_start

    @property
    def not_in_same_cycle(self) -> Dict[str, List[str]]:
        """
        Gates that cannote start in the same cycle.
        """

        return self._not_in_same_cycle

    @staticmethod
    def _check_machine_properties_mapping(
        machine_properties: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if the given machine properties mapping is a valid descriptions of the
        machine properties and returns a Dict to easily initialize a MachineProperties
        object.

        :param machine_properties: Mapping containing the machine properties.
        :raises ValueError: If there are missing keys in the Mapping.
        :return: Dict to easily initialize a MachineProperties object.
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
