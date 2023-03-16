"""This module contains the ``MachineProperties`` class, which helps with conveniently
setting up machine properties for the routing environment.

Usage:
    The code below will create ``MachineProperties`` which will have the following
    properties:

    * The machine has two qubits.
    * The machine supports the X, Y, C-NOT and Measure gates.


    .. code-block:: python

        from qgym.envs.routing import MachineProperties

        hardware_spec = MachineProperties(n_qubits=2)
        hardware_spec.add_gates({"x": 2, "y": 2, "cnot": 4, "measure": 10})


"""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Set, Tuple, Union

from qgym.utils import GateEncoder
from qgym.utils.input_validation import check_instance, check_int, check_string


class MachineProperties:
    """``MachineProperties`` is a class to conveniently setup machine properties for the
    ``Routing`` environment.
    """

    def __init__(self, n_qubits: int) -> None:
        """
        Init of the MachineProperties class.

        :param n_qubits: Number of qubits of the machine.
        """
        self._n_qubits = check_int(n_qubits, "n_qubits", l_bound=1)
        self._gates: Dict[Any, int] = {}

    @classmethod
    def from_file(cls, filename: str) -> MachineProperties:
        """Load MachineProperties from a JSON file. Not implemented."""
        raise NotImplementedError(
            "Loading machine properties from files is not yet implemented."
        )

    def add_gates(self, gates: Mapping[str, int]) -> MachineProperties:
        """Add gates to the machine properties that should be supported.

        :param gates: ``Mapping`` of gates that the machine can perform as keys, and the
            number of machine cycles (time) as values.
        :return: The ``MachineProperties`` with the added gates.
        """
        check_instance(gates, "gates", Mapping)

        for gate_name, n_cycles in gates.items():
            gate_name = check_string(gate_name, "gate name", lower=True)
            if gate_name in self._gates:
                msg = f"Gate '{gate_name}' was already given. Overwriting it with the "
                msg += "new value."
                warnings.warn(msg)
            self._gates[gate_name] = n_cycles
        return self

    def encode(self) -> GateEncoder:
        """Encode the gates in the machine properties to integer values.

        :return: The GateEncoder used to encode the gates. This GateEncoder can be used
            to decode the gates or encode quantum circuits containing the same gate
            names as in this ``MachineProperties`` object.
        """
        if (
            any(not isinstance(gate, str) for gate in self._gates)
        ):
            msg = "Gate names of machine properties are not of type str, perhaps they "
            msg = "are already encoded?"
            raise ValueError(msg)

        gate_encoder = GateEncoder().learn_gates(self._gates)

        self._gates = gate_encoder.encode_gates(self._gates)
        return gate_encoder

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits of the machine."""
        return self._n_qubits

    @property
    def gates(self) -> Union[Dict[str, int], Dict[int, int]]:
        """Return a``Dict`` with the gate names the machine can perform as keys, and the
        number of machine cycles (time) as values.
        """
        return self._gates

    @property
    def n_gates(self) -> int:
        """Return the number of supported gates."""
        return len(self._gates)

    def __str__(self) -> str:
        """Make a string representation of the machine properties."""
        text = f"{self.__class__.__name__}:\n"
        text += f"n_qubits: {self._n_qubits}\n"
        text += f"gates: {self._gates}"
        return text

    def __repr__(self) -> str:
        """Make a string representation without endline characters."""
        text = f"{self.__class__.__name__}("
        text += f"n_qubits={self._n_qubits}, "
        text += f"gates={self._gates})"
        return text
