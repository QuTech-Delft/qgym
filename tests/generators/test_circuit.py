"""This module contains tests for the graph generation module."""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx
import pytest

from qgym.custom_types import Gate
from qgym.generators.circuit import (
    BasicCircuitGenerator,
    CircuitGenerator,
    NullCircuitGenerator,
    WorkshopCircuitGenerator,
)


def test_abc() -> None:
    with pytest.raises(TypeError):
        CircuitGenerator()


class TestNullCircuitGenerator:

    @pytest.fixture(name="generator")
    def null_graph_generator_fixture(self) -> None:
        return NullCircuitGenerator()

    def test_infinite(self, generator: NullCircuitGenerator) -> None:
        assert not generator.finite

    def test_inheritance(self, generator: NullCircuitGenerator) -> None:
        assert isinstance(generator, CircuitGenerator)
        assert isinstance(generator, Iterator)

    def test_next(self, generator: NullCircuitGenerator) -> None:
        circuit = next(generator)
        assert isinstance(circuit, list)
        assert len(circuit) == 0

    def test_iter(self, generator: NullCircuitGenerator) -> None:
        for i, circuit in enumerate(generator):
            assert isinstance(circuit, list)
            assert len(circuit) == 0

            if i > 100:
                break


class TestBasicCircuitGenerator:

    def __init__(self) -> None:
        self.n_qubits = 5
        self.max_length = 50

    @pytest.fixture(name="simple_generator")
    def null_graph_generator_fixture(self) -> None:
        return BasicCircuitGenerator(self.n_qubits, self.max_length, seed=42)

    def test_infinite(self, simple_generator: BasicCircuitGenerator) -> None:
        assert not simple_generator.finite

    def test_inheritance(self, generator: BasicCircuitGenerator) -> None:
        assert isinstance(generator, CircuitGenerator)
        assert isinstance(generator, Iterator)

    def test_next(self, simple_generator: BasicCircuitGenerator) -> None:
        circuit = next(simple_generator)
        assert isinstance(circuit, list)
        assert self.n_qubits <= len(circuit) <= self.max_length
        self._check_circuit(circuit)

    def test_iter(self, simple_generator: BasicCircuitGenerator) -> None:
        for i, circuit in enumerate(simple_generator):
            assert isinstance(circuit, list)
            assert self.n_qubits <= len(circuit) <= self.max_length
            self._check_circuit(circuit)

            if i > 100:
                break

    def _check_circuit(self, circuit: list[Gate]) -> None:
        for i in range(self.n_qubits):
            gate = circuit[i]
            assert isinstance(gate, Gate)
            assert gate.name == "prep"
            assert gate.q1 == i
            assert gate.q1 == i

        for i in range(self.n_qubits, len(circuit)):
            gate = circuit[i]
            assert isinstance(gate, Gate)
            assert gate.name in {"x", "y", "z", "cnot", "measure"}
            if gate.name == "cnot":
                assert gate.q1 != gate.q2
            else:
                assert gate.q1 == gate.q2

    def test_seed(self) -> None:
        generator1 = BasicCircuitGenerator(10, seed=1)
        generator2 = BasicCircuitGenerator(10, seed=1)
        generator3 = BasicCircuitGenerator(10, seed=3)

        for _ in range(10):
            curcuit1 = next(generator1)
            curcuit2 = next(generator2)
            curcuit3 = next(generator3)

            assert curcuit1 == curcuit2
            assert curcuit1 != curcuit3


class TestWorkshopCircuitGenerator:

    def __init__(self) -> None:
        self.n_qubits = 5
        self.max_length = 50

    @pytest.fixture(name="simple_generator")
    def null_graph_generator_fixture(self) -> None:
        return WorkshopCircuitGenerator(self.n_qubits, self.max_length, seed=42)

    def test_infinite(self, simple_generator: WorkshopCircuitGenerator) -> None:
        assert not simple_generator.finite

    def test_inheritance(self, generator: WorkshopCircuitGenerator) -> None:
        assert isinstance(generator, CircuitGenerator)
        assert isinstance(generator, Iterator)

    def test_next(self, simple_generator: WorkshopCircuitGenerator) -> None:
        circuit = next(simple_generator)
        assert isinstance(circuit, list)
        assert self.n_qubits <= len(circuit) <= self.max_length
        self._check_circuit(circuit)

    def test_iter(self, simple_generator: WorkshopCircuitGenerator) -> None:
        for i, circuit in enumerate(simple_generator):
            assert isinstance(circuit, list)
            assert self.n_qubits <= len(circuit) <= self.max_length
            self._check_circuit(circuit)

            if i > 100:
                break

    def _check_circuit(self, circuit: list[Gate]) -> None:
        for i in range(self.n_qubits, len(circuit)):
            gate = circuit[i]
            assert isinstance(gate, Gate)
            assert gate.name in {"x", "y", "cnot", "measure"}
            if gate.name == "cnot":
                assert gate.q1 != gate.q2
            else:
                assert gate.q1 == gate.q2

    def test_seed(self) -> None:
        generator1 = BasicCircuitGenerator(10, seed=1)
        generator2 = BasicCircuitGenerator(10, seed=1)
        generator3 = BasicCircuitGenerator(10, seed=3)

        for _ in range(10):
            curcuit1 = next(generator1)
            curcuit2 = next(generator2)
            curcuit3 = next(generator3)

            assert curcuit1 == curcuit2
            assert curcuit1 != curcuit3
