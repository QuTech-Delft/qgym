"""Metrics to assess the performance of InitialMappers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import networkx as nx
import numpy as np

from qgym.benchmarks.benchmark_result import BenchmarkResult
from qgym.utils import Circuit, CircuitLike
from qgym.utils.input_parsing import parse_connection_graph

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import ArrayLike

    from qgym.templates.pass_protocols import Mapper

# pylint: disable=too-few-public-methods


@runtime_checkable
class InitialMappingMetric(Protocol):
    """Protocol that a metric for initial mapping should follow."""

    @abstractmethod
    def compute(self, circuit: CircuitLike, mapping: ArrayLike) -> float:
        """Compute the metric for the provided `circuit` and `mapping`."""


class DistanceRatioLoss(InitialMappingMetric):
    """The :class:`DistanceRatioLoss` class."""

    def __init__(self, connection_graph: nx.Graph | ArrayLike) -> None:
        """Init of the :class:`DistanceRatioLoss` class.

        Args:
            connection_graph: :class:`networkx.Graph` representation of the QPU
                topology. Each node represents a physical qubit and each edge represents
                a connection in the QPU topology.

        """
        self.connection_graph = parse_connection_graph(connection_graph)
        n_qubits = len(self.connection_graph)
        self.distance_matrix = np.zeros((n_qubits, n_qubits), dtype=np.int_)
        for node_u, distances in nx.all_pairs_shortest_path_length(
            self.connection_graph
        ):
            for node_v in self.connection_graph:
                self.distance_matrix[node_u, node_v] = distances[node_v]

    def compute(self, circuit: CircuitLike, mapping: ArrayLike) -> float:
        """Compute the distance ratio loss.

        Args:
            circuit: Quantum circuit to compute the metric for.
            mapping: 1D-ArrayLike representing a qubit mapping.

        Returns:
            The distance ratio loss.
        """
        circuit = Circuit(circuit)
        interaction_graph = circuit.get_interaction_graph()
        if interaction_graph.number_of_edges() == 0:
            return 1.0
        mapping = np.asarray(mapping, dtype=np.int_)
        distance_loss = 0
        for edge in interaction_graph.edges():
            mapped_edge = (mapping[edge[0]], mapping[edge[1]])
            distance_loss += self.distance_matrix[mapped_edge]

        return float(distance_loss / interaction_graph.number_of_edges())


class InitialMappingBenchmarker:
    """The :class:`InitialMappingBenchmarker` class."""

    def __init__(
        self,
        generator: Iterator[CircuitLike],
        metrics: Iterable[InitialMappingMetric],
    ) -> None:
        """Init of the :class:`InitialMappingBenchmarker` class.

        Args:
            generator: Circuit generator. Currently only the ``MaxCutQAOAGenerator`` is
                supported.
            metrics: Metrics to compute.
        """
        self.generator = generator
        self.metrics = tuple(metrics)

    def run(self, mapper: Mapper, max_iter: int = 1000) -> BenchmarkResult:
        """Run the benchmark.

        Args:
            mapper: Mapper to benchmark.
            max_iter: Maximum number of iterations to benchmark.

        Returns:
            :class:`~qgym.benchmarks.metrics.BenchmarkResult` containing the results
            from the benchmark.
        """
        results: list[deque[float]] = [deque() for _ in self.metrics]
        for i, circuit in enumerate(self.generator, start=1):
            mapping = mapper.compute_mapping(circuit)
            for metric, result_que in zip(self.metrics, results):
                result_que.append(metric.compute(circuit, mapping))

            if i >= max_iter:
                break

        return BenchmarkResult(results)
