"""
Generic utils for the Quantum RL Gym
"""
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from numpy.typing import NDArray


def check_adjacency_matrix(adjacency_matrix: NDArray[Any]) -> None:
    """
    :param adjacency_matrix: Matrix to check.
    :raise ValueError: In case the provided input is not a valid matrix.
    """
    if (
        not adjacency_matrix.ndim == 2
        and adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    ):
        raise ValueError("The provided value should be a square 2-D adjacency matrix.")


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
        for idx, gate_name in enumerate(gates):
            if gate_name in self.encoding_dct:
                # Don't save duplicate names
                idx -= 1
            else:
                self.encoding_dct[gate_name] = idx
                self.decoding_dct[idx] = gate_name
                self.longest_name = max(self.longest_name, len(gate_name))
        self.n_gates = idx + 1

        return self

    def encode_gates(
        self, gates: Union[str, Mapping[str, Any], Sequence[Tuple[str, int, int]]]
    ) -> Union[int, Dict[int, Any], List[Tuple[int, int, int]]]:
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
            for gate_name in gates:
                gate_encoding = self.encoding_dct[gate_name]
                encoded_gates[gate_encoding] = deepcopy(gates[gate_name])

        elif isinstance(gates, Sequence):
            encoded_gates = []
            for gate_name, control_qubit, target_qubit in gates:
                gate_encoding = self.encoding_dct[gate_name]
                encoded_gates.append((gate_encoding, control_qubit, target_qubit))

        else:
            raise TypeError(
                f"gates type must be str, Mapping or Sequence, got {type(gates)}."
            )

        return encoded_gates

    def decode_gates(
        self,
        encoded_gates: Union[int, Mapping[int, Any], Sequence[Tuple[int, int, int]]],
    ) -> Union[str, Dict[str, Any], List[Tuple[str, int, int]]]:
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
            for gate_int, control_qubit, target_qubit in encoded_gates:
                decoded_gate = self.decoding_dct[gate_int]
                decoded_gates.append((decoded_gate, control_qubit, target_qubit))

        else:
            raise TypeError(
                "encoded_gates must be int, Mapping or Sequence, got "
                f"{type(encoded_gates)}."
            )

        return decoded_gates
