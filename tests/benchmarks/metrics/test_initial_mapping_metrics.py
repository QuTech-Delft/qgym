from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from qgym.benchmarks import (
    BenchmarkResult,
    DistanceRatioLoss,
    InitialMappingBenchmarker,
)
from qgym.generators import MaxCutQAOAGenerator


@pytest.fixture(name="connection_graph")
def connection_graph_fixture() -> nx.Graph:
    return nx.from_edgelist([(0, 1), (0, 2), (0, 3), (0, 4)])


def circuit1() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.cx((0, 0, 0, 0, 1), (1, 2, 3, 4, 2))
    return circuit


def circuit2() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.cx((0, 1, 2, 3), (1, 2, 3, 0))
    return circuit


@pytest.mark.parametrize(
    ("circuit", "ratio_loss"), [(circuit1(), 6 / 5), (circuit2(), 1.5)]
)
def test_distance_ratio_loss(
    connection_graph: nx.Graph, circuit: QuantumCircuit, ratio_loss: float
) -> None:
    quality_metric = DistanceRatioLoss(connection_graph)
    result = quality_metric.compute(circuit, np.arange(5))
    assert result == ratio_loss


def test_initial_mapping_metric() -> None:
    metric = DistanceRatioLoss(nx.cycle_graph(4))
    generator = MaxCutQAOAGenerator(4, 0.5, seed=42)
    benchmarker = InitialMappingBenchmarker(metrics=[metric], generator=generator)

    class SimpleMapper:
        connection_graph = nx.cycle_graph(4)

        def compute_mapping(self, interaction_graph: nx.Graph) -> NDArray[np.int_]:
            return np.arange(len(interaction_graph))

    mapper = SimpleMapper()
    result = benchmarker.run(mapper, max_iter=500)
    assert isinstance(result, BenchmarkResult)
    assert result.raw_data.shape == (1, 500)
    np.testing.assert_array_equal(1, result.raw_data >= 1)
