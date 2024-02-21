"""This module contains graph generators for :class:`~qgym.envs.Routing`."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import SupportsInt

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_int


class InteractionGenerator(
    Iterator[NDArray[np.int_]]
):  # pylint: disable=too-few-public-methods
    """Abstract Base Class for interaction circuit generation.

    All interaction circuit generators should inherit from :class:`InteractionGenerator`
    to be compatible with the :class:`~qgym.envs.Routing` environemnt.
    """

    finite: bool
    """Boolean value stating wether the generator is finite."""

    @abstractmethod
    def __next__(self) -> NDArray[np.int_]:
        """Make a new interaction circuit.

        The __next__ method of a :class:`InteractionGenerator` should generate a NDArray
        of shape (len_circuit, 2) with dtype int. Each pair represent he indices of two
        qubits that have an interaction in the circuit.
        """


class BasicInteractionGenerator(InteractionGenerator):
    """:class:`BasicInteractionGenerator` is an interaction generation implementation.

    Interactions are completely randomly generated using the ``numpy`` `choice`_ method.

    .. _choice: https://numpy.org/doc/stable/reference/random/generated/
       numpy.random.choice.html
    """

    def __init__(
        self,
        n_qubits: SupportsInt,
        max_length: SupportsInt = 10,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`BasicInteractionGenerator`.

        Args:
            n_qubits: Number of qubits.
            max_length: Maximum length of the generated interaction circuits. Defaults
                to 10.
            seed: Seed to use.
        """
        self.n_qubits = check_int(n_qubits, "n_qubits", l_bound=1)
        self.max_length = check_int(max_length, "max_length", l_bound=1)
        self.rng = parse_seed(seed)
        self.finite = False
        super().__init__()

    def __repr__(self) -> str:
        """String representation of the :class:`BasicInteractionGenerator`."""
        return (
            f"BasicInteractionGenerator[max_length={self.max_length}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> NDArray[np.int_]:
        """Create a new randomly generated graph."""
        length = self.rng.integers(1, self.max_length + 1)

        circuit = np.zeros((length, 2), dtype=int)
        for idx in range(length):
            circuit[idx] = self.rng.choice(self.n_qubits, size=2, replace=False)

        return circuit


class NullInteractionGenerator(InteractionGenerator):
    """Generator class for generating empty interaction circuits.

    Useful for unit testing.
    """

    def __init__(self) -> None:
        """Init of the :class:`NullInteractionGenerator`"""
        self.finite = False
        super().__init__()

    def __next__(self) -> NDArray[np.int_]:
        """Create a new empty interaction circuit."""
        return np.empty((0, 2), dtype=int)

    def __repr__(self) -> str:
        """String representation of the :class:`NullInteractionGenerator`."""
        return f"NullInteractionGenerator[finite={self.finite}]"
