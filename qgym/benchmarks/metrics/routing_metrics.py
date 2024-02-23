"""Metrics to assess the performance of Routers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

import networkx as nx
from qiskit import QuantumCircuit  ##TODO: find other option than qiskit dependency.
import numpy as np
from numpy.typing import NDArray


class RoutingSolutionQuality:

    def __init__(
        self,
        circuit: QuantumCircuit,
    ) -> None:
        # pylint: disable=line-too-long
        """Init of the :class:`~qgym.benchmarks.metrics.routing.RoutingSolutionQuality` class.

        Args:
            connection_graph: `networkx Graph <https://networkx.org/documentation/stable/reference/classes/graph.html>`_
                representation of the QPU topology. Each node represents a physical
                qubit and each edge represents a connection in the QPU topology.
        """

    def gates_ratio_loss(
        self,
        swaps_added: int,
    ):

        gates_after_routing = self.circuit.count_ops() + swaps_added
        return gates_after_routing / self.circuit.count_ops()
