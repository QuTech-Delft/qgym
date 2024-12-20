"""This module contains protocols for different compilation passes."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from qgym.utils import Circuit, CircuitLike

# pylint: disable=too-few-public-methods


@runtime_checkable
class Mapper(Protocol):
    """Mapper protocol."""

    @abstractmethod
    def compute_mapping(self, circuit: CircuitLike) -> NDArray[np.int_]:
        """Compute a mapping for a provided quantum `circuit`."""


@runtime_checkable
class Router(Protocol):
    """Qubit router protocol."""

    @abstractmethod
    def compute_routing(self, circuit: CircuitLike) -> Circuit:
        """Compute a qubit routing for a provided quantum `circuit`."""
