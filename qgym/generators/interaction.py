"""This module contains interaction generators for :class:`~qgym.envs.Routing`."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator, SupportsInt

import networkx as nx
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_graph_is_valid_topology, check_int


class InteractionGenerator(Iterator[NDArray[np.int_]]):
    """Abstract Base Class for interaction circuit generation.

    All interaction circuit generators should inherit from :class:`InteractionGenerator`
    to be compatible with the :class:`~qgym.envs.Routing` environment.
    """

    finite: bool
    """Boolean value stating whether the generator is finite."""

    @abstractmethod
    def __next__(self) -> NDArray[np.int_]:
        """Make a new interaction circuit.

        The ``__next__`` method of a :class:`InteractionGenerator` should generate a
        :class:`~numpy.ndarray` of shape (len_circuit, 2) with dtype ``int``. Each pair
        represent the indices of two qubits that have an interaction in the circuit.
        """

    @abstractmethod
    def set_state_attributes(self, **kwargs: Any) -> None:
        """Set attributes that the state can receive.

        This method is called inside the scheduling environment to receive information
        about the state. The same keywords as for the the init of the
        :class:`~qgym.envs.routing.RoutingState` are provided.
        """


class BasicInteractionGenerator(InteractionGenerator):
    """:class:`BasicInteractionGenerator` is an interaction generation implementation.

    Interactions are completely randomly generated using the :func:`numpy.random.choice`
    method.
    """

    def __init__(
        self,
        max_length: SupportsInt = 10,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`BasicInteractionGenerator`.

        Args:
            max_length: Maximum length of the generated interaction circuits. Defaults
                to 10.
            seed: Seed to use.
        """
        self.max_length = check_int(max_length, "max_length", l_bound=1)
        self.rng = parse_seed(seed)
        self.finite = False
        self.n_qubits: int
        super().__init__()

    def set_state_attributes(self, **kwargs: dict[str, Any]) -> None:
        """Set the `n_qubits` attribute.

        Args:
            kwargs: Keyword arguments. Must have the ``"connection_graph"`` key with a
                :class:`~networkx.Graph` representation of the connection graph.
        """
        connection_graph: nx.Graph = kwargs["connection_graph"]
        check_graph_is_valid_topology(connection_graph, "connection_graph")
        self.n_qubits = connection_graph.number_of_nodes()

    def __repr__(self) -> str:
        """String representation of the :class:`BasicInteractionGenerator`."""
        return (
            f"BasicInteractionGenerator[max_length={self.max_length}, "
            f"rng={self.rng}, "
            f"finite={self.finite}]"
        )

    def __next__(self) -> NDArray[np.int_]:
        """Create a new randomly generated interaction circuit."""
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

    def set_state_attributes(self, **kwargs: dict[str, Any]) -> None:
        """Receive state attributes, but do nothing with it.

        Args:
            kwargs: Keyword arguments.
        """
