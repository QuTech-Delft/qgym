"""
This module contains the GateEncoder class encoding gate to integers and back.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

from qgym.custom_types import Gate


class GateEncoder:
    """
    Learns a set of gates and creates a mapping to integers and back.
    """

    def __init__(self) -> None:
        """
        Initialize the `GateEncoder`.
        """
        self.n_gates = 0

    def learn_gates(self, gates: Iterable) -> GateEncoder:
        """
        Learns the gates names from and Iterable and creates a mapping from unique gate
        names to integers and back.

        :param gates: An iterable containing the names of the gates that must be
            learned.
        """
        self._encoding_dct = {}
        self._decoding_dct = {}
        self._longest_name = 0

        idx = 0  # in case gates is empty
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

    def encode_gates(
        self,
        gates: Union[str, Mapping[str, Any], Sequence[Gate], Iterable[str]],
    ) -> Union[int, Dict[int, Any], List[Gate]]:
        """
        Encodes gate names in gates to integers, based on the gates seen in
        `learn_gates`.

        :param gates: gate name(s) to encode
        :return: integer encoded version of gates
        :raise TypeError: When an unsupported type is given
        """
        if isinstance(gates, str):
            encoded_gates = self._encoding_dct[gates]

        elif isinstance(gates, Mapping):
            encoded_gates = {}
            for gate_name, item in gates.items():
                gate_encoding = self._encoding_dct[gate_name]
                if isinstance(item, int):
                    encoded_gates[gate_encoding] = item
                elif isinstance(item, Iterable):
                    item_encoded = []
                    for i in item:
                        item_encoded.append(self._encoding_dct[i])
                    encoded_gates[gate_encoding] = item_encoded
                else:
                    raise ValueError("Unknown mapping")

        elif isinstance(gates, Sequence) and isinstance(gates[0], Gate):
            encoded_gates = []
            for gate in gates:
                encoded_name = self._encoding_dct[gate.name]
                encoded_gates.append(Gate(encoded_name, gate.q1, gate.q2))

        elif isinstance(gates, Iterable):
            encoded_gates = []
            for gate_name in gates:
                gate_encoding = self._encoding_dct[gate_name]
                encoded_gates.append(gate_encoding)

        else:
            raise TypeError(
                f"gates type must be str, Mapping or Sequence, got {type(gates)}."
            )

        return encoded_gates

    def decode_gates(
        self,
        encoded_gates: Union[int, Mapping[int, Any], Sequence[Gate], Iterable[int]],
    ) -> Union[str, Dict[str, Any], List[Gate]]:
        """
        Decodes integer encoded gate names to the original gate names based on the gates
        seen in the learn_gates function.

        :param encoded_gates: Encoded gates that are to be decoded.
        :return: Decoded version of encoded_gates.
        :raise TypeError: When an unsupported type is given
        """
        if isinstance(encoded_gates, int):
            decoded_gates = self._decoding_dct[encoded_gates]

        elif isinstance(encoded_gates, Mapping):
            decoded_gates = {}
            for gate_int in encoded_gates:
                gate_name = self._decoding_dct[gate_int]
                decoded_gates[gate_name] = encoded_gates[gate_int]

        elif isinstance(encoded_gates, Sequence) and isinstance(encoded_gates[0], Gate):
            decoded_gates = []
            for gate in encoded_gates:
                decoded_gate_name = self._decoding_dct[gate.name]
                decoded_gates.append(Gate(decoded_gate_name, gate.q1, gate.q2))

        elif isinstance(encoded_gates, Iterable):
            decoded_gates = []
            for gate_int in encoded_gates:
                decoded_gate = self._decoding_dct[gate_int]
                decoded_gates.append(decoded_gate)

        else:
            raise TypeError(
                "encoded_gates must be int, Mapping or Sequence, got "
                f"{type(encoded_gates)}."
            )

        return decoded_gates
