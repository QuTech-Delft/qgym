"""
This module contains the GateEncoder class which encoded gate to integers and back.
"""
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

from qgym._custom_types import Gate


class GateEncoder:
    """
    Learns a set of gates and creates a mapping to integers and back.
    """

    def learn_gates(self, gates: Iterable) -> "GateEncoder":
        """
        Learns the gates names from and Iterable and creates a mapping from unique gate
        names to integers and back.
        :gates: An Iterable containing the names of the gates which must be learned
        """
        self.encoding_dct = {}
        self.decoding_dct = {}
        self.longest_name = 0
        for idx, gate_name in enumerate(gates, 1):
            self.encoding_dct[gate_name] = idx
            self.decoding_dct[idx] = gate_name
            self.longest_name = max(self.longest_name, len(gate_name))
        self.n_gates = idx

        return self

    def encode_gates(
        self,
        gates: Union[str, Mapping[str, Any], Sequence[Gate], Iterable[str]],
    ) -> Union[int, Dict[int, Any], List[Gate]]:
        """
        Encodes gate names in gates to integers, based on the gates seen in the
        learn_gates function.
        :param gates: gates to encode
        :return: integer encoded version of gates
        :raise TypeError: In case that an unsoprted type is given
        """
        if isinstance(gates, str):
            encoded_gates = self.encoding_dct[gates]

        elif isinstance(gates, Mapping):
            encoded_gates = {}
            for gate_name, item in gates.items():
                gate_encoding = self.encoding_dct[gate_name]
                if isinstance(item, int):
                    encoded_gates[gate_encoding] = item
                elif isinstance(item, Iterable):
                    item_encoded = []
                    for i in item:
                        item_encoded.append(self.encoding_dct[i])
                    encoded_gates[gate_encoding] = item_encoded
                else:
                    raise ValueError("Unkown mapping")

        elif isinstance(gates, Sequence):
            encoded_gates = []
            for gate in gates:
                encoded_name = self.encoding_dct[gate.name]
                encoded_gates.append(Gate(encoded_name, gate.q1, gate.q2))

        elif isinstance(gates, Iterable):
            encoded_gates = []
            for gate_name in gates:
                gate_encoding = self.encoding_dct[gate_name]
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
        Decodes int encoded gate names to the original gate names based on the gates
        seen in the learn_gates function.
        :param encoded_gates: gates to decode
        :return: decoded version of encoded_gates
        :raise TypeError: In case that an unsoprted type is given
        """
        if isinstance(encoded_gates, int):
            decoded_gates = self.decoding_dct[encoded_gates]

        elif isinstance(encoded_gates, Mapping):
            decoded_gates = {}
            for gate_int in encoded_gates:
                gate_name = self.decoding_dct[gate_int]
                decoded_gates[gate_name] = encoded_gates[gate_int]

        elif isinstance(encoded_gates, Sequence):
            decoded_gates = []
            for gate in encoded_gates:
                decoded_gatename = self.decoding_dct[gate.name]
                decoded_gates.append(Gate(decoded_gatename, gate.q1, gate.q2))

        elif isinstance(encoded_gates, Iterable):
            decoded_gates = []
            for gate_int in encoded_gates:
                decoded_gate = self.decoding_dct[gate_int]
                decoded_gates.append(decoded_gate)

        else:
            raise TypeError(
                "encoded_gates must be int, Mapping or Sequence, got "
                f"{type(encoded_gates)}."
            )

        return decoded_gates
