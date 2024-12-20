"""This module contains circuit generators for :class:`~qgym.envs.Scheduling`."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, SupportsInt

import numpy as np

from qgym.custom_types import Gate
from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_int

if TYPE_CHECKING:
    from numpy.random import Generator


class CircuitGenerator(Iterator[list[Gate]]):
    """Abstract Base Class for circuit generation used for scheduling.

    All interaction circuit generators should inherit from :class:`CircuitGenerator`
    to be compatible with the :class:`~qgym.envs.Scheduling` environment.
    """

    finite: bool
    """Boolean value stating whether the generator is finite."""

    @abstractmethod
    def __next__(self) -> list[Gate]:
        """Make a new circuit.

        The __next__ method of a :class:`CircuitGenerator` should generate a list of
        Gates.

        Example circuit:
            >>> circuit = [Gate("prep", 0,0), Gate("prep", 1,1), Gate("cnot", 0,1)]
        """

    @abstractmethod
    def set_state_attributes(self, **kwargs: Any) -> None:
        """Set attributes that the state can receive.

        This method is called inside the scheduling environment to receive information
        about the state. The same keywords as for the the init of the
        :class:`~qgym.envs.scheduling.SchedulingState` are provided.
        """


class BasicCircuitGenerator(CircuitGenerator):
    """:class:`BasicCircuitGenerator` is a basic random circuit generator."""

    def __init__(self, seed: Generator | SupportsInt | None = None) -> None:
        """Init of the :class:`BasicInteractionGenerator`.

        Args:
            seed: Seed to use.
        """
        self.rng = parse_seed(seed)
        self.finite = False
        self.n_qubits: int
        self.max_gates: int

    def set_state_attributes(
        self, machine_properties: Any = None, max_gates: SupportsInt = 0, **kwargs: Any
    ) -> None:
        """Set the `n_qubits` and `max_gates` attributes.

        Args:
            machine_properties: :class:`~qgym.envs.scheduling.MachineProperties`
                containing at least the number of qubits of the machine.
            max_gates: Maximum number of gates allowed in the circuit.
            kwargs: Additional keyword arguments. These are not used.
        """
        if not hasattr(machine_properties, "n_qubits"):
            msg = "'machine_properties' did not have the 'n_qubits' attribute"
            raise AttributeError(msg)
        self.n_qubits = machine_properties.n_qubits
        self.max_gates = check_int(max_gates, "max_gates", l_bound=1)

    def __repr__(self) -> str:
        """String representation of the :class:`BasicCircuitGenerator`."""
        return (
            f"BasicCircuitGenerator[n_qubits={self.n_qubits} "
            f"max_gates={self.max_gates}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> list[Gate]:
        """Create a new randomly generated circuit.

        The length of the circuit is a random integer in the interval
        [`n_qubits`, `max_length`].
        """
        n_gates = self.rng.integers(self.n_qubits, self.max_gates, endpoint=True)
        gate_names = ["x", "y", "z", "cnot", "measure"]
        probabilities = [0.16, 0.16, 0.16, 0.5, 0.02]

        circuit = [Gate("prep", qubit, qubit) for qubit in range(self.n_qubits)]

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
    """:class:`WorkshopCircuitGenerator` is a simplified random circuit generator."""

    def __init__(self, seed: Generator | SupportsInt | None = None) -> None:
        """Init of the :class:`WorkshopCircuitGenerator`.

        Args:
            seed: Seed to use.
        """
        self.rng = parse_seed(seed)
        self.finite = False
        self.n_qubits: int
        self.max_gates: int

    def set_state_attributes(self, **kwargs: dict[str, Any]) -> None:
        """Set the- `n_qubits` and `max_gates` attributes.

        Args:
            kwargs: Keyword arguments. Must have the keys ``"machine_properties"`` and
                ``"max_gates"`` with values of type
                :class:`~qgym.envs.scheduling.MachineProperties` and integerlike
                (``SupportsInt``) respectively.
        """
        machine_properties = kwargs["machine_properties"]
        if not hasattr(machine_properties, "n_qubits"):
            msg = "'machine_properties' did not have the 'n_qubits' attribute"
            raise AttributeError(msg)
        self.n_qubits = machine_properties.n_qubits
        self.max_gates = check_int(kwargs["max_gates"], "max_gates", l_bound=1)

    def __repr__(self) -> str:
        """String representation of the :class:`WorkshopCircuitGenerator`."""
        return (
            f"WorkshopCircuitGenerator[n_qubits={self.n_qubits} "
            f"max_gates={self.max_gates}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> list[Gate]:
        """Create a new randomly generated circuit.

        The length of the circuit is a random integer in the interval
        [`n_qubits`, `max_length`].
        """
        n_gates = self.rng.integers(self.n_qubits, self.max_gates, endpoint=True)
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
        """Init of the :class:`NullCircuitGenerator` class."""
        self.finite = False

    def __next__(self) -> list[Gate]:
        """Create a new empty circuit."""
        circuit: list[Gate] = []
        return circuit

    def __repr__(self) -> str:
        """String representation of the :class:`NullCircuitGenerator`."""
        return f"NullCircuitGenerator[finite={self.finite}]"

    def set_state_attributes(self, **kwargs: dict[str, Any]) -> None:
        """Receive state attributes, but do nothing with it.

        Args:
            kwargs: Keyword arguments.
        """
