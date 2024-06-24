from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import ArrayLike

from qgym.benchmarks import DistanceRatioLoss


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
    [(small_graph(), 6 / 5)],
)
def test_distance_ratio_loss(
    smallest_graph: nx.Graph, interaction_graph: nx.Graph, ratio_loss: float
) -> None:
    quality_metric = DistanceRatioLoss(connection_graph=smallest_graph)
    result = quality_metric.compute(
        interaction_graph=interaction_graph, mapping=[0, 1, 2, 3, 4]
    )

    assert result == ratio_loss
