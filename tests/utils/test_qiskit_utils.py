from __future__ import annotations

import itertools
from collections.abc import Hashable

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout, TrivialLayout, VF2Layout

from qgym.utils.qiskit_utils import (
    QiskitMapperWrapper,
    _get_qreg_to_int_mapping,
    _parse_circuit,
    get_interaction_graph,
)


class TestParseCircuit:

    def test_quantum_circuit(self) -> None:
        circuit = QuantumCircuit(1)
        dag_circuit = _parse_circuit(circuit)
        assert circuit is not dag_circuit
        assert dag_circuit.qregs == {"q": QuantumRegister(1, "q")}
        assert len([x for x in dag_circuit.nodes()]) == 2

    def test_dag_circuit(self) -> None:
        dag_circuit = DAGCircuit()
        assert dag_circuit is _parse_circuit(dag_circuit)


@pytest.mark.parametrize(
    ("circuit", "expected_output"),
    [
        (
            QuantumCircuit(2),
            {
                Qubit(QuantumRegister(2, "q"), 0): 0,
                Qubit(QuantumRegister(2, "q"), 1): 1,
            },
        ),
        (
            QuantumCircuit(QuantumRegister(2, "q")),
            {
                Qubit(QuantumRegister(2, "q"), 0): 0,
                Qubit(QuantumRegister(2, "q"), 1): 1,
            },
        ),
        (
            QuantumCircuit(QuantumRegister(2, "q"), QuantumRegister(1, "a")),
            {
                Qubit(QuantumRegister(2, "q"), 0): 0,
                Qubit(QuantumRegister(2, "q"), 1): 1,
                Qubit(QuantumRegister(1, "a"), 0): 2,
            },
        ),
    ],
)
def test_get_qreg_to_int_mapping(
    circuit: QuantumCircuit, expected_output: dict[Hashable, int]
) -> None:
    assert _get_qreg_to_int_mapping(circuit) == expected_output


class TestGetInteractionGraph:

    @pytest.mark.parametrize("circuit_size", range(6))
    def test_empty_circuit(self, circuit_size: int) -> None:
        circuit = QuantumCircuit(circuit_size)
        interaction_graph = get_interaction_graph(circuit)
        assert nx.is_isomorphic(interaction_graph, nx.empty_graph(circuit_size))

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_dense_circuit(self, circuit_size: int) -> None:
        circuit = QuantumCircuit(circuit_size)
        circuit.h(range(circuit_size))
        circuit.cx(*zip(*itertools.combinations(range(circuit_size), 2)))
        interaction_graph = get_interaction_graph(circuit)
        assert nx.is_isomorphic(interaction_graph, nx.complete_graph(circuit_size))

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_line_circuit(self, circuit_size: int) -> None:
        circuit = QuantumCircuit(circuit_size)
        circuit.cx(range(circuit_size - 1), range(1, circuit_size))
        interaction_graph = get_interaction_graph(circuit)
        assert nx.is_isomorphic(interaction_graph, nx.path_graph(circuit_size))

    def test_multi_qubit_value_error(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.ccx(0, 1, 2)
        with pytest.raises(ValueError, match="no 3\+ qubit operations are supported"):
            get_interaction_graph(circuit)


class TestQiskitMapperWrapper:

    @pytest.fixture(name="circuit")
    def circuit_fixture(self) -> QuantumCircuit:
        """Create a circuit with a cycle of cx gates."""
        circuit = QuantumCircuit(6)
        circuit.cx(list(range(6)), [*range(1, 6), 0])
        return circuit

    @pytest.fixture(name="coupling_map")
    def coupling_map_fixture(self) -> CouplingMap:
        """Create a coupling map for a 6 qubit device with a cycle topology."""
        return CouplingMap(zip(range(6), [*range(1, 6), 0]))

    def test_trivial_layout(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_mapper = TrivialLayout(coupling_map)
        mapper = QiskitMapperWrapper(qiskit_mapper)
        mapping = mapper.compute_mapping(circuit)
        np.testing.assert_equal(mapping, range(6))

    def test_vf2_layout(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_mapper = VF2Layout(coupling_map)
        mapper = QiskitMapperWrapper(qiskit_mapper)
        mapping = mapper.compute_mapping(circuit)
        optimal_mapping_found = False
        for i in range(6):
            opt_map1 = np.roll(np.arange(6), i)
            opt_map2 = np.flip(opt_map1)
            if np.array_equal(mapping, opt_map1) or np.array_equal(mapping, opt_map2):
                optimal_mapping_found = True
                break
        assert optimal_mapping_found

    def test_sabre_layout(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_mapper = SabreLayout(coupling_map)
        mapper = QiskitMapperWrapper(qiskit_mapper)
        mapper.compute_mapping(circuit)
