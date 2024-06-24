"""Metrics to assess the performance of InitialMappers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from typing import Literal, Protocol, runtime_checkable

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from qgym.generators.graph import BasicGraphGenerator, GraphGenerator
from qgym.utils.input_validation import check_string


@runtime_checkable
class InitialMappingMetric(Protocol):
    def compute(self, interaction_graph: nx.Graph, mapping: ArrayLike) -> float: ...


@runtime_checkable
class Mapper(Protocol):
    def compute_mapping(self, interaction_graph: nx.Graph) -> NDArray[np.int_]: ...


class DistanceRatioLoss(InitialMappingMetric):

    def __init__(self, connection_graph: nx.Graph) -> None:
        """Init of the :class:`InitialMappingSolutionQuality` class.

        Args:
            connection_graph: :class:`networkx.Graph` representation of the QPU
                topology. Each node represents a physical qubit and each edge represents
                a connection in the QPU topology.

        """
        self.connection_graph = connection_graph
        n_qubits = len(self.connection_graph)
        self.distance_matrix = np.zeros((n_qubits, n_qubits), dtype=np.int_)
        for node_u, distances in nx.all_pairs_shortest_path_length(connection_graph):
            for node_v in connection_graph:
                self.distance_matrix[node_u, node_v] = distances[node_v]

    def compute(self, interaction_graph: nx.Graph, mapping: ArrayLike) -> float:
        if interaction_graph.number_of_edges() == 0:
            return 1.0
        mapping = np.asarray(mapping, dtype=np.int_)
        distance_loss = 0
        for edge in interaction_graph.edges():
            mapped_edge = (mapping[edge[0]], mapping[edge[1]])
            distance_loss += self.distance_matrix[mapped_edge]

        return distance_loss / interaction_graph.number_of_edges()


class AgentPerformance:

    def __init__(
        self,
    ) -> None:
        """Init of the :class:`AgentPerformance` class.

        Args:
        """


class InitialMappingBenchmarker:

    def __init__(
        self,
        generator: GraphGenerator | None = None,
        *,
        metrics: Iterable[InitialMappingMetric] | InitialMappingMetric,
    ) -> None:
        """Init of the :clas:`InitialMappingBenchmarker` class.

        Args:
            generator: Interaction graph generator to use during benchmarking
            metrics: Metrics to compute.
        """
        self.generator = BasicGraphGenerator() if generator is None else generator
        self.metrics = (
            (metrics,) if isinstance(metrics, InitialMappingMetric) else tuple(metrics)
        )
        for metric in self.metrics:
            if hasattr(metric, "connection_graph"):
                connection_graph = deepcopy(metric.connection_graph)
                self.generator.set_state_attributes(connection_graph=connection_graph)
                break

    def run(
        self,
        mapper: Mapper,
        max_iter: int = 1000,
        return_type: Literal["raw", "quartiles", "median", "mean"] = "raw",
    ) -> NDArray[np.float_]:
        """Run the benchmark.

        Args:
            mapper: Mapper to benchmark.
            max_iter: Maximum number of iterations to benchmark.
            return_type: Return type to return the results.

        Returns:
            NDArray containing the results from the benchmark.
        """
        return_type = check_string(return_type, "return_type", lower=True)

        results = [deque() for _ in self.metrics]
        for i, interaction_graph in enumerate(self.generator, start=1):
            mapping = mapper.compute_mapping(interaction_graph)
            for metric, result_que in zip(self.metrics, results):
                result_que.append(metric.compute(interaction_graph, mapping))

            if i >= max_iter:
                break

        if return_type == "raw":
            return np.array(results, dtype=np.float_)
        if return_type == "quartiles":
            return np.quantile(results, [0, 0.25, 0.5, 0.75, 1], axis=1)
        if return_type == "median":
            return np.median(results, axis=1)
        if return_type == "mean":
            return np.mean(results, axis=1)
