"""This module contains the ``GateEncoder`` class which can be used for encoding gates
to integers and back.

Usage:
    >>> from qgym.utils import GateEncoder
    >>> encoder = GateEncoder().learn_gates(["x", "y", "z", "cnot", "measure"])
    >>> encoded_list = encoder.encode_gates(["x", "x", "measure", "z"])
    >>> print(encoded_list)
    [1, 1, 5, 3]
    >>> encoder.decode_gates(encoded_list)
    ['x', 'x', 'measure', 'z']
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping, Sequence, TypeVar, cast, overload

from qgym.custom_types import Gate

T = TypeVar("T")


class GateEncoder:
    """Learns a set of gates and creates a mapping to integers and back."""

    def __init__(self) -> None:
        """Initialize the ``GateEncoder``."""
        self.n_gates = 0
        self._encoding_dct: dict[str, int] = {}
        self._decoding_dct: dict[int, str] = {}
        self._longest_name = 0

    def learn_gates(self, gates: Iterable[str]) -> GateEncoder:
        """Learns the gates names from an ``Iterable`` and creates a mapping from unique
        gate names to integers and back.

        :param gates: An ``Iterable`` containing the names of the gates that should be
            learned. The ``Iterable`` can contain duplicate names.
        :returns: Self.
        """
        self.n_gates = 0
        for idx, gate_name in enumerate(gates, 1):
            if gate_name in self._encoding_dct:
                warnings.warn(f"'gates' contains multiple entries of {gate_name}")
            else:
                self._encoding_dct[gate_name] = idx
                self._decoding_dct[idx] = gate_name
                self._longest_name = max(self._longest_name, len(gate_name))
                self.n_gates += 1

        return self

    @overload
    def encode_gates(self, gates: str) -> int:
        ...

    @overload
    def encode_gates(self, gates: Mapping[str, T]) -> dict[int, T]:
        ...

    @overload
    def encode_gates(self, gates: Sequence[Gate]) -> list[Gate]:
        ...

    @overload
    def encode_gates(self, gates: set[str]) -> set[int]:
        ...

    @overload
    def encode_gates(self, gates: list[str] | tuple[str, ...]) -> list[int]:
        ...

    def encode_gates(
        self,
        gates: str
        | Mapping[str, Any]
        | Sequence[Gate]
        | set[str]
        | list[str]
        | tuple[str, ...],
    ) -> int | dict[int, Any] | list[Gate] | set[int] | list[int]:
        """Encode the gate names (of type ``str``) in `gates` to integers, based on the
        gates seen in ``learn_gates``.

        :param gates: Gates to encode. The input type determines the return type.
        :return: Integer encoded version of gates. The output structure should resemble
            the input structure. So a ``Mapping`` will return a ``Dict``, while a single
            ``str`` will return an ``int``.
        :raise TypeError: When an unsupported type is given.
        """
        if isinstance(gates, str):
            encoded_str = self._encoding_dct[gates]
            return encoded_str

        if isinstance(gates, Mapping):
            return self._encode_mapping(gates)

        if isinstance(gates, Sequence) and isinstance(gates[0], Gate):
            # We assume that if the first element of gates is a Gate, then the whole
            # Sequence contains Gate objects.
            encoded_gates_list: list[Gate] = []
            for gate in cast(Sequence[Gate], gates):
                encoded_name = self._encoding_dct[gate.name]
                encoded_gates_list.append(Gate(encoded_name, gate.q1, gate.q2))
            return encoded_gates_list

        if isinstance(gates, set):
            encoded_names_set: set[int] = set()
            for gate_name in gates:
                gate_encoding = self._encoding_dct[gate_name]
                encoded_names_set.add(gate_encoding)
            return encoded_names_set

        if isinstance(gates, (list, tuple)):
            encoded_names_list: list[int] = []
            for gate_name in gates:
                gate_encoding = self._encoding_dct[gate_name]
                encoded_names_list.append(gate_encoding)
            return encoded_names_list

        raise TypeError(
            f"gates type must be str, Mapping or Sequence, got {type(gates)}."
        )

    def _encode_mapping(self, mapping: Mapping[str, Any]) -> dict[int, Any]:
        """Encode a mapping with gate names.

        :raise ValueError: For unknown mappings.
        """
        encoded_dict: dict[int, Any] = {}
        gate_name: str
        for gate_name, item in mapping.items():
            gate_encoding = self._encoding_dct[gate_name]
            if isinstance(item, int):
                encoded_dict[gate_encoding] = item
            elif isinstance(item, Iterable):
                item_encoded = []
                for i in item:
                    item_encoded.append(self._encoding_dct[i])
                encoded_dict[gate_encoding] = item_encoded
            else:
                raise ValueError("Unknown mapping")
        return encoded_dict

    @overload
    def decode_gates(self, encoded_gates: int) -> str:
        ...

    @overload
    def decode_gates(self, encoded_gates: Mapping[int, Any]) -> dict[str, Any]:
        ...

    @overload
    def decode_gates(self, encoded_gates: Sequence[Gate]) -> list[Gate]:
        ...

    @overload
    def decode_gates(self, encoded_gates: set[int]) -> set[str]:
        ...

    @overload
    def decode_gates(self, encoded_gates: list[int] | tuple[int, ...]) -> list[str]:
        ...

    def decode_gates(
        self,
        encoded_gates: int
        | Mapping[int, Any]
        | Sequence[Gate]
        | set[int]
        | list[int]
        | tuple[int, ...],
    ) -> str | dict[str, Any] | list[Gate] | set[str] | list[str]:
        """Decode integer encoded gate names to the original gate names based on the
        gates seen in ``learn_gates``.

        :param encoded_gates: Encoded gates that are to be decoded. The input type
            determines the return type.
        :return: Decoded version of encoded_gates. The output structure should resemble
            the input structure. So a ``Mapping`` will return a ``Dict``, while a single
            ``int`` will return a ``str``.
        :raise TypeError: When an unsupported type is given
        """
        if isinstance(encoded_gates, int):
            decoded_int = self._decoding_dct[encoded_gates]
            return decoded_int

        if isinstance(encoded_gates, Mapping):
            decoded_dict: dict[str, Any] = {}
            gate_int: int
            for gate_int in encoded_gates:
                gate_name = self._decoding_dct[gate_int]
                decoded_dict[gate_name] = encoded_gates[gate_int]
            return decoded_dict

        if isinstance(encoded_gates, Sequence) and isinstance(encoded_gates[0], Gate):
            # We assume that if the first element of encoded_gates is a Gate, then the
            # whole Sequence contains Gate objects.
            decoded_gate_list: list[Gate] = []
            for gate in cast(Sequence[Gate], encoded_gates):
                decoded_gate_name = self._decoding_dct[gate.name]
                decoded_gate_list.append(Gate(decoded_gate_name, gate.q1, gate.q2))
            return decoded_gate_list

        if isinstance(encoded_gates, set):
            decoded_name_set: set[str] = set()
            for gate_int in encoded_gates:
                decoded_gate = self._decoding_dct[gate_int]
                decoded_name_set.add(decoded_gate)
            return decoded_name_set

        if isinstance(encoded_gates, (list, tuple)):
            decoded_name_list: list[str] = []
            for gate_int in encoded_gates:
                decoded_gate = self._decoding_dct[gate_int]
                decoded_name_list.append(decoded_gate)
            return decoded_name_list

        raise TypeError(
            "encoded_gates must be int, Mapping or Sequence, got "
            f"{type(encoded_gates)}."
        )

    def __repr__(self) -> str:
        """Make a string representation without endline characters."""
        return f"{self.__class__.__name__}(encoding={self._encoding_dct})"
