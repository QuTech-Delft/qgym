"""This module contains a class that can generate random circuits"""
from typing import List, Tuple, Union, Optional
from numbers import Integral

import numpy as np
from numpy.random import Generator, default_rng


class RandomCircuitGenerator:
    """Generates random circuits in the form of a list of tuples.
    ex format. [("prep", 0,0), ("prep", 1,1), ("cnot", 0,1)]"""

    def __init__(
        self, n_qubits: Integral, max_gates: Integral, rng: Optional[Generator] = None
    ) -> None:
        self.n_qubits = n_qubits
        self.max_gates = max_gates
        self._rng = rng

    @property
    def rng(self) -> Generator:
        """The random number generator of this circuit generator. If none is set yet,
        this will generate a new one, with a random seed."""
        if self._rng is None:
            self._rng = default_rng()
        return self._rng

    @rng.setter
    def rng(self, rng: Generator) -> None:
        self._rng = rng

    def generate_circuit(
        self, n_gates: Union[str, Integral] = "random"
    ) -> List[Tuple[str, int, int]]:
        """Make a random circuit with prep, measure, x, y, z, and cnot operations

        :param n_gates: If "random", then a circuit of random length will be made, if
            an int a circuit of length min(n_gates, max_gates) will be made.
        :return: A randomly generated circuit"""

        if n_gates.lower().strip() == "random":
            n_gates = self.rng.integers(self.n_qubits, self.max_gates, endpoint=True)
        else:
            n_gates = min(n_gates, self.max_gates)

        circuit = [None] * n_gates

        # Every circuit should start by initializing the qubits
        for qubit in range(self.n_qubits):
            circuit[qubit] = ("prep", qubit, qubit)

        gates = ["x", "y", "z", "cnot", "measure"]
        p = [0.16, 0.16, 0.16, 0.5, 0.02]
        for idx in range(self.n_qubits, n_gates):
            gate = self.rng.choice(gates, p=p)

            if gate == "cnot":
                control_qubit, target_qubit = self.rng.choice(
                    np.arange(self.n_qubits), size=2, replace=False
                )
            else:
                control_qubit = self.rng.integers(self.n_qubits)
                target_qubit = control_qubit

            circuit[idx] = (gate, control_qubit, target_qubit)

        return circuit
