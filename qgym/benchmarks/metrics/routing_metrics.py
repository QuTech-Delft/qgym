"""Metrics to assess the performance of Routers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Protocol, SupportsInt, runtime_checkable

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

from qgym.benchmarks.benchmark_result import BenchmarkResult
from qgym.templates.pass_protocols import Router
from qgym.utils.input_validation import check_int
from qgym.utils.qiskit_utils import parse_circuit


# pylint: disable=too-few-public-methods
@runtime_checkable
class RoutingMetric(Protocol):
    """Protocol that a metric for qubit routing should follow."""

    @abstractmethod
    def compute(
        self,
        input_circuit: QuantumCircuit | DAGCircuit,
        routed_circuit: QuantumCircuit | DAGCircuit,
    ) -> float:
        """Compute the metric for the provided `input_circuit` and `routed_circuit`."""


class InteractionRatioLoss(RoutingMetric):
    """The :class:`InteractionRatioLoss` class."""

    def __init__(self, swap_penalty: SupportsInt = 3) -> None:
        """Init of :class:`InteractionRatioLoss`.

        Args:
            swap_penalty: Number of gates to use to decompose the SWAP gate. Since a
                SWAP gate is often decomposed using 3 CNOT gates, the default is 3.
        """
        self.swap_penalty = check_int(swap_penalty, "swap_penalty", l_bound=1)

    def compute(
        self,
        input_circuit: QuantumCircuit | DAGCircuit,
        routed_circuit: QuantumCircuit | DAGCircuit,
    ) -> float:
        """Method to calculate the ratio of the input and output circuit.

        The ratio loss is defined by the number of 2 qubit gates in the `output_circuit`
        devided by the number of 2 qubit gates in the `input_circut`. A score of 1 is
        thus a perfect score and the lowest value that can be returned.

        Args:
            input_circuit: Input circuit before routing was performed.
            routed_circut: Routed version of the input circuit.

        Returns:
            The routing solution quality ratio.
        """
        input_n_iteractions = self.get_num_interactions(input_circuit)
        routed_n_iteractions = self.get_num_interactions(routed_circuit)
        try:
            return float(routed_n_iteractions / input_n_iteractions)
        except ZeroDivisionError:
            if not routed_n_iteractions:
                return 1.0
            return float("inf")

    def get_num_interactions(self, circuit: QuantumCircuit | DAGCircuit) -> int:
        """Get the number of interactions from the `circuit`.

        Args:
            circuit: Circuit to count the number of interactions from.
            swap_penalty: Number of gates to use to decompose the SWAP gate. Since a
                SWAP gate is often decomposed using 3 CNOT gates, the default is 3.

        Returns:
            Number of 2 qubit interactions, where each SWAP gate is counted
            `swap_penalty` times.
        """
        dag = parse_circuit(circuit)
        return sum(
            self.swap_penalty if gate.name == "swap" else 1
            for gate in dag.two_qubit_ops()
        )


class RoutingBenchmarker:
    """The :class:`RoutingBenchmarker` class."""

    def __init__(
        self,
        generator: Iterator[QuantumCircuit],
        metrics: Iterable[RoutingMetric],
    ) -> None:
        """Init of the :class:`RoutingBenchmarker` class.

        Args:
            generator: Circuit generator. Currently only the ``MaxCutQAOAGenerator`` is
                supported.
            metrics: Metrics to compute.
        """
        self.generator = generator
        self.metrics = tuple(metrics)

    def run(self, router: Router, max_iter: int = 1000) -> BenchmarkResult:
        """Run the benchmark.

        Args:
            router: Router to benchmark.
            max_iter: Maximum number of iterations to benchmark.

        Returns:
            :class:`~qgym.benchmarks.metrics.BenchmarkResult` containing the results
            from the benchmark.
        """
        results: list[deque[float]] = [deque() for _ in self.metrics]
        for i, circuit in enumerate(self.generator, start=1):
            routed_circuit = router.compute_routing(circuit)
            for metric, result_que in zip(self.metrics, results):
                result_que.append(metric.compute(circuit, routed_circuit))

            if i >= max_iter:
                break

        return BenchmarkResult(results)
