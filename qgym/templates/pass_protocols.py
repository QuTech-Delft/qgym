"""This module contains protocols for different compilation passes."""

from __future__ import annotations
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

# pylint: disable=too-few-public-methods


@runtime_checkable
class Mapper(Protocol):
    """Mapper protocol."""

    @abstractmethod
    def compute_mapping(self, circuit: QuantumCircuit | DAGCircuit) -> NDArray[np.int_]:
        """Compute a mapping for a provided quantum `circuit`."""
