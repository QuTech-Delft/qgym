"""This module contains the ``RandomCircuitGenerator``, which can be used to generate
random circuits. Here, a quantum circuit is a ``List`` with ``Gate`` objects.
"""
from typing import List, Optional, Union

import numpy as np
from numpy.random import Generator, default_rng

from qgym.custom_types import Gate


class RandomCircuitGenerator:
    """Generates random circuits in the form of a ``List`` with ``Gate`` objects.

    Example circuit:
        >>> circuit = [Gate("prep", 0,0), Gate("prep", 1,1), Gate("cnot", 0,1)]
    """

    def __init__(
        self, n_qubits: int, max_gates: int, rng: Optional[Generator] = None
    ) -> None:
        """Initialize the ``RandomCircuitGenerator``.

        :param n_qubits: Number of qubits of the circuit.
        :param max_gates: Maximum number of gates that a circuit may have.
        :param rng: Optional random number generator.
        """
        self.n_qubits = n_qubits
        self.max_gates = max_gates
        self._rng = rng

    @property
    def rng(self) -> Generator:
        """Return the random number generator of this ``RandomCircuitGenerator``. If
        none is set yet, this will generate a new one, using
        ``numpy.random.default_rng``.

        :return: Random number generator.
        """
        if self._rng is None:
            self._rng = default_rng()
        return self._rng

    @rng.setter
    def rng(self, rng: Generator) -> None:
        self._rng = rng

    def generate_circuit(
        self, n_gates: Union[str, int] = "random", mode: str = "default"
    ) -> List[Gate]:
        """Generate a random quantum circuit.

        :param n_gates: If "random", then a circuit of random length will be made. If
            an ``int`` is given, a circuit of length ``min(n_gates, max_gates)`` will
            be made.
        :param mode: If mode is "default", a circuit will be generated containing the
            'prep', 'x', 'y', 'z', 'cnot' and 'measure' gates. If mode is "workshop",
            a simpler circuit containing just 'h', 'cnot' and `measure` gates will be
            generated.
        :return: A randomly generated quantum circuit.
        """
        if isinstance(n_gates, str):
            if n_gates.lower().strip() == "random":
                n_gates = self.rng.integers(
                    self.n_qubits, self.max_gates, endpoint=True
                )
            else:
                raise ValueError(f"Unknown flag {n_gates}, choose from 'random'.")
        elif isinstance(n_gates, int):
            n_gates = min(n_gates, self.max_gates)
        else:
            msg = "n_gates should be of type int or str, but was of type "
            msg += f"{type(n_gates)}."
            raise ValueError(msg)

        circuit: List[Gate] = [Gate("", -1, -1)] * n_gates

        if mode.lower() == "default":
            gate_names = ["x", "y", "z", "cnot", "measure"]
            p = [0.16, 0.16, 0.16, 0.5, 0.02]
        elif mode.lower() == "workshop":
            gate_names = ["x", "y", "cnot", "measure"]
            p = [0.2, 0.2, 0.5, 0.1]
        else:
            raise ValueError("Unknown mode, choose 'default' or 'workshop'.")

        for idx in range(n_gates):
            name = self.rng.choice(gate_names, p=p)

            if name == "cnot":
                q1, q2 = self.rng.choice(
                    np.arange(self.n_qubits), size=2, replace=False
                )
            else:
                q1 = self.rng.integers(self.n_qubits)
                q2 = q1

            circuit[idx] = Gate(name, q1, q2)

        # If mode is default, the circuit should start by initializing the qubits
        if mode.lower() == "default":
            for qubit in range(self.n_qubits):
                circuit[qubit] = Gate("prep", qubit, qubit)

        return circuit
