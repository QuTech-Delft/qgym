from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from qgym.benchmarks import (
    BenchmarkResult,
    DistanceRatioLoss,
    InitialMappingBenchmarker,
)


@pytest.fixture
def smallest_graph() -> nx.Graph:
    smallest_graph = nx.Graph()
    smallest_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    return smallest_graph


def small_graph() -> nx.Graph:
    small_graph = nx.Graph()
    small_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)])
    return small_graph


@pytest.mark.parametrize(
    "interaction_graph, ratio_loss",
    [(small_graph(), 6 / 5), (nx.cycle_graph(4), 1.5)],
)
def test_distance_ratio_loss(
    smallest_graph: nx.Graph, interaction_graph: nx.Graph, ratio_loss: float
) -> None:
    quality_metric = DistanceRatioLoss(connection_graph=smallest_graph)
    result = quality_metric.compute(
        interaction_graph=interaction_graph, mapping=np.arange(5)
    )

    assert result == ratio_loss


def test_initial_mapping_metric(smallest_graph: nx.Graph) -> None:
    metric = DistanceRatioLoss(nx.cycle_graph(4))
    benchmarker = InitialMappingBenchmarker(metrics=metric)

    class SimpleMapper:
        connection_graph = smallest_graph

        def compute_mapping(self, interaction_graph: nx.Graph) -> NDArray[np.int_]:
            return np.arange(len(interaction_graph))

    mapper = SimpleMapper()
    result = benchmarker.run(mapper, max_iter=500)
    assert isinstance(result, BenchmarkResult)
    assert result.raw_data.shape == (1, 500)
    np.testing.assert_array_equal(1, result.raw_data >= 1)
