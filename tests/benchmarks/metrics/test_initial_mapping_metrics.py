from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import ArrayLike

from qgym.benchmarks.metrics.initial_mapping_metrics import InitialMappingSolutionQuality


@pytest.fixture
def smallest_graph() -> nx.Graph:
    smallest_graph = nx.Graph()
    smallest_graph.add_edge(0, 1)
    smallest_graph.add_edge(0, 2)
    smallest_graph.add_edge(0, 3)
    smallest_graph.add_edge(0, 4)
    return smallest_graph


def small_graph() -> nx.Graph:
    small_graph = nx.Graph()
    small_graph.add_edge(0, 1)
    small_graph.add_edge(0, 2)
    small_graph.add_edge(0, 3)
    small_graph.add_edge(0, 4)
    small_graph.add_edge(1, 2)
    return small_graph


@pytest.mark.parametrize(
    "interaction_graph, ratio_loss",
    [(small_graph(), 6 / 5), (small_graph().add_edge(2, 3), 8 / 6)],
)
def test_distance_ratio_loss(interaction_graph: nx.Graph(), ratio_loss: float) -> None:
    quality_metric = InitialMappingSolutionQuality(connection_graph=smallest_graph())
    result = quality_metric.distance_ratio_loss(
        interaction_graph=interaction_graph, mapping=[0, 1, 2, 3, 4]
    )

    assert result == ratio_loss
