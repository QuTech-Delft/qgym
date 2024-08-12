from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from numpy.typing import ArrayLike

"""
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qgym.benchmarks.metrics.routing_metrics import RoutingSolutionQuality


@pytest.fixture
def circuit_1() -> QuantumCircuit:
    qr = QuantumRegister(3, "q")
    anc = QuantumRegister(1, "ancilla")
    cr = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qr, anc, cr)

    qc.x(anc[0])
    qc.h(anc[0])
    qc.h(qr[0:3])
    qc.cx(qr[0:3], anc[0])
    qc.h(qr[0:3])
    qc.barrier(qr)
    qc.measure(qr, cr)
    return qc


def test_distance_ratio_loss() -> None:
    quality_metric = RoutingSolutionQuality(circuit=circuit_1)
    # TODO
    initial_number_of_gates = circuit_1.count_ops()
    swaps_added = 2
    loss = quality_metric.gates_ratio_loss(swaps_added=swaps_added)

    assert loss == (initial_number_of_gates + swaps_added) / initial_number_of_gates
"""
