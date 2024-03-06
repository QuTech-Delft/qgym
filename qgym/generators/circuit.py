"""This module contains graph generators for :class:`~qgym.envs.Routing`."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import SupportsInt

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from qgym.custom_types import Gate
from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_int


class CircuitGenerator(
    Iterator[NDArray[np.int_]]
):  # pylint: disable=too-few-public-methods
    """Abstract Base Class for circuit generation used for scheduling.

    All interaction circuit generators should inherit from :class:`CircuitGenerator`
    to be compatible with the :class:`~qgym.envs.Scheduling` environemnt.
    """

    finite: bool
    """Boolean value stating wether the generator is finite."""

    @abstractmethod
    def __next__(self) -> list[Gate]:
        """Make a new circuit.

        The __next__ method of a :class:`CircuitGenerator` should generate a list of
        Gates.

        Example circuit:
            >>> circuit = [Gate("prep", 0,0), Gate("prep", 1,1), Gate("cnot", 0,1)]
        """


class BasicCircuitGenerator(CircuitGenerator):
    """:class:`BasicCircuitGenerator` is a basic random circuit generation
    implementation.
    """

    def __init__(
        self,
        n_qubits: SupportsInt,
        max_length: SupportsInt = 50,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`BasicInteractionGenerator`.

        Args:
            n_qubits: Number of qubits.
            max_length: Maximum length of the generated circuits. Defaults to 50.
            seed: Seed to use.
        """
        self.n_qubits = check_int(n_qubits, "n_qubits", l_bound=1)
        self.max_length = check_int(max_length, "max_length", l_bound=1)
        self.rng = parse_seed(seed)
        self.finite = False

    def __repr__(self) -> str:
        """String representation of the :class:`BasicCircuitGenerator`."""
        return (
            f"BasicCircuitGenerator[n_qubits={self.n_qubits} "
            f"max_length={self.max_length}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> list[Gate]:
        """Create a new randomly generated circuit.

        The length of the circuit is a random integer in the interval
        [`n_qubits`, `max_length`].
        """
        n_gates = self.rng.integers(self.n_qubits, self.max_length, endpoint=True)
        gate_names = ["x", "y", "z", "cnot", "measure"]
        probabilities = [0.16, 0.16, 0.16, 0.5, 0.02]

        circuit: list[Gate] = []

        for qubit in range(self.n_qubits):
            circuit.append(Gate("prep", qubit, qubit))

        for _ in range(self.n_qubits, n_gates):
            name = self.rng.choice(gate_names, p=probabilities)

            if name == "cnot":
                qubit1, qubit2 = self.rng.choice(
                    np.arange(self.n_qubits), size=2, replace=False
                )
            else:
                qubit1 = self.rng.integers(self.n_qubits)
                qubit2 = qubit1

            circuit.append(Gate(name, qubit1, qubit2))

        return circuit


class WorkshopCircuitGenerator(CircuitGenerator):
    """:class:`WorkshopCircuitGenerator` is a simplified random circuit generation
    implementation.
    """

    def __init__(
        self,
        n_qubits: SupportsInt,
        max_length: SupportsInt = 10,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`WorkshopCircuitGenerator`.

        Args:
            n_qubits: Number of qubits.
            max_length: Maximum length of the generated circuits. Defaults to 10.
            seed: Seed to use.
        """
        self.n_qubits = check_int(n_qubits, "n_qubits", l_bound=1)
        self.max_length = check_int(max_length, "max_length", l_bound=1)
        self.rng = parse_seed(seed)
        self.finite = False

    def __repr__(self) -> str:
        """String representation of the :class:`WorkshopCircuitGenerator`."""
        return (
            f"WorkshopCircuitGenerator[n_qubits={self.n_qubits} "
            f"max_length={self.max_length}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> list[Gate]:
        """Create a new randomly generated circuit.

        The length of the circuit is a random integer in the interval
        [`n_qubits`, `max_length`].
        """
        n_gates = self.rng.integers(self.n_qubits, self.max_length, endpoint=True)
        gate_names = ["x", "y", "cnot", "measure"]
        probabilities = [0.2, 0.2, 0.5, 0.1]

        circuit: list[Gate] = []

        for _ in range(n_gates):
            name = self.rng.choice(gate_names, p=probabilities)

            if name == "cnot":
                qubit1, qubit2 = self.rng.choice(
                    np.arange(self.n_qubits), size=2, replace=False
                )
            else:
                qubit1 = self.rng.integers(self.n_qubits)
                qubit2 = qubit1

            circuit.append(Gate(name, qubit1, qubit2))

        return circuit


class NullCircuitGenerator(CircuitGenerator):
    """Generator class for generating empty circuits.

    Useful for unit testing.
    """

    def __init__(self) -> None:
        """Init of the :class:`NullCircuitGenerator`"""
        self.finite = False

    def __next__(self) -> list[Gate]:
        """Create a new empty circuit."""
        circuit: list[Gate] = []
        return circuit

    def __repr__(self) -> str:
        """String representation of the :class:`NullCircuitGenerator`."""
        return f"NullCircuitGenerator[finite={self.finite}]"
