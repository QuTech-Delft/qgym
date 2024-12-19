from __future__ import annotations
from qgym.benchmarks.metrics import (
    RoutingMetric,
    InteractionRatioLoss,
    RoutingBenchmarker,
)
import numpy as np
from qgym.generators import MaxCutQAOAGenerator
import pytest
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qgym.utils.qiskit_utils import parse_circuit
from qgym.benchmarks import BenchmarkResult


def circuit1() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.cx((0, 0, 0, 0, 1), (1, 2, 3, 4, 2))
    return circuit


def circuit1_routed() -> QuantumCircuit:
    # Routed on a ring topology
    circuit = QuantumCircuit(5)
    circuit.cx((0, 0, 1), (1, 4, 2))
    circuit.swap(0, 1)
    circuit.cx(1, 2)
    circuit.swap(1, 2)
    circuit.cx(2, 3)
    return circuit


def circuit2() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.x(range(4))
    return circuit


def circuit2_badly_routed() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.x(range(4))
    circuit.swap(1, 2)
    return circuit


class TestInteractionRatioLoss:
    @pytest.fixture(name="metric")
    def metric_fixture(self) -> InteractionRatioLoss:
        return InteractionRatioLoss()

    def test_instance_routing_metric(self, metric: InteractionRatioLoss) -> None:
        assert isinstance(metric, RoutingMetric)

    @pytest.mark.parametrize(
        ("circuit", "n_interactions"),
        [
            (circuit1(), 5),
            (circuit1_routed(), 11),
            (circuit2(), 0),
            (circuit2_badly_routed(), 3),
        ],
    )
    def test_get_num_interactions(
        self, metric: InteractionRatioLoss, circuit: QuantumCircuit, n_interactions: int
    ) -> None:
        assert metric.get_num_interactions(circuit) == n_interactions

    @pytest.mark.parametrize(
        ("circuit", "routed_circuit", "loss"),
        [
            (circuit1(), circuit1_routed(), 2.2),
            (circuit2(), circuit2(), 1),
            (circuit2(), circuit2_badly_routed(), float("inf")),
        ],
    )
    def test_compute(
        self,
        metric: InteractionRatioLoss,
        circuit: QuantumCircuit,
        routed_circuit: int,
        loss: float,
    ) -> None:
        assert metric.compute(circuit, routed_circuit) == loss


def test_routing_benchmarker() -> None:
    metric = InteractionRatioLoss()
    generator = MaxCutQAOAGenerator(4, 0.5, seed=42)
    benchmarker = RoutingBenchmarker(metrics=[metric], generator=generator)

    class SimpleRouter:
        def compute_routing(self, circuit: QuantumCircuit | DAGCircuit) -> DAGCircuit:
            return parse_circuit(circuit)

    router = SimpleRouter()
    result = benchmarker.run(router, max_iter=500)
    assert isinstance(result, BenchmarkResult)
    assert result.raw_data.shape == (1, 500)
    np.testing.assert_array_equal(1, result.raw_data >= 1)
