from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from qgym.utils.qiskit_utils import (
    get_interaction_circuit,
    get_interaction_graph,
    insert_swaps_in_circuit,
    parse_circuit,
)


class TestParseCircuit:
    def test_quantum_circuit(self) -> None:
        circuit = QuantumCircuit(1)
        dag_circuit = parse_circuit(circuit)
        assert circuit is not dag_circuit
        assert dag_circuit.qregs == {"q": QuantumRegister(1, "q")}
        assert len([x for x in dag_circuit.nodes()]) == 2

    def test_dag_circuit(self) -> None:
        dag_circuit = DAGCircuit()
        assert dag_circuit is parse_circuit(dag_circuit)


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


class TestGetInteractionCircuit:
    @pytest.mark.parametrize("circuit_size", range(1, 6))
    def test_empty_circuit(self, circuit_size: int) -> None:
        circuit = QuantumCircuit(circuit_size)
        interaction_circuit = get_interaction_circuit(circuit)
        np.testing.assert_array_equal(interaction_circuit, np.empty((0, 2)))

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_dense_circuit(self, circuit_size: int) -> None:
        circuit = QuantumCircuit(circuit_size)
        circuit.h(range(circuit_size))
        circuit.cx(*zip(*itertools.combinations(range(circuit_size), 2)))
        circuit.h(range(circuit_size))

        expected_result = np.array([*itertools.combinations(range(circuit_size), 2)])
        interaction_circuit = get_interaction_circuit(circuit)
        np.testing.assert_array_equal(interaction_circuit, expected_result)

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_line_circuit(self, circuit_size: int) -> None:
        circuit = QuantumCircuit(circuit_size)
        circuit.h(range(circuit_size))
        circuit.cx(range(circuit_size - 1), range(1, circuit_size))
        circuit.h(range(circuit_size))

        expected_result = np.array(
            [np.arange(circuit_size - 1), np.arange(1, circuit_size)]
        ).T
        interaction_circuit = get_interaction_circuit(circuit)
        np.testing.assert_array_equal(interaction_circuit, expected_result)

    def test_multi_qubit_value_error(self) -> None:
        circuit = QuantumCircuit(3)
        circuit.ccx(0, 1, 2)
        with pytest.raises(ValueError, match="no 3\+ qubit operations are supported"):
            get_interaction_graph(circuit)

    @pytest.mark.parametrize(
        "circuit",
        [
            QuantumCircuit(),
            QuantumCircuit(QuantumRegister(1, "q"), QuantumRegister(1, "a")),
            QuantumCircuit(QuantumRegister(1, "a")),
        ],
        ids=["no registers", "multiple registers", "wrong name"],
    )
    def test_non_physical_circuit_error(self, circuit: QuantumCircuit) -> None:
        expected_msg = "Interaction circuits are defined for physical circuits only"
        with pytest.raises(ValueError, match=expected_msg):
            get_interaction_circuit(circuit)


class TestAddSwapsToCircuit:
    def test_multiple_swaps(self) -> None:
        """
        Input:                              Expected Output:
        q_0: ──■─────────────────           q_0: ──■────────────────────
             ┌─┴─┐                               ┌─┴─┐             ┌───┐
        q_1: ┤ X ├──■────────────           q_1: ┤ X ├─────■────X──┤ X ├
             └───┘┌─┴─┐                          └───┘     │    │  └─┬─┘
        q_2: ─────┤ X ├──■───────           q_2: ──────X───┼────X────┼──
                  └───┘┌─┴─┐                           │ ┌─┴─┐       │
        q_3: ──────────┤ X ├──■──           q_3: ──X───X─┤ X ├──■────┼──
                       └───┘┌─┴─┐                  │     └───┘┌─┴─┐  │
        q_4: ───────────────┤ X ├           q_4: ──X──────────┤ X ├──■──
                            └───┘                             └───┘
        """
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)

        swaps = [(1, 3, 4), (1, 2, 3), (3, 1, 2)]

        expected_circuit = QuantumCircuit(5)
        expected_circuit.cx(0, 1)
        expected_circuit.swap(3, 4)
        expected_circuit.swap(2, 3)
        expected_circuit.cx(1, 3)
        expected_circuit.swap(1, 2)
        expected_circuit.cx(3, 4)
        expected_circuit.cx(4, 1)

        routed_dag = insert_swaps_in_circuit(circuit, swaps)
        assert expected_circuit == dag_to_circuit(routed_dag)

    def test_preserve_single_qubit_gates(self) -> None:
        """
        Input:                              Expected Output:
             ┌───┐                               ┌───┐             ┌───┐┌───┐
        q_0: ┤ H ├──■─────────────────      q_0: ┤ H ├──■────────X─┤ X ├┤ H ├
             └───┘┌─┴─┐┌───┐                     └───┘┌─┴─┐┌───┐ │ └─┬─┘└───┘
        q_1: ─────┤ X ├┤ H ├──■───────      q_1: ─────┤ X ├┤ H ├─┼───■───────
                  └───┘└───┘┌─┴─┐┌───┐                └───┘└───┘ │
        q_2: ───────────────┤ X ├┤ H ├      q_2: ────────────────X───────────
                            └───┘└───┘
        """
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.h(2)

        swaps = [(1, 0, 2)]

        expected_circuit = QuantumCircuit(3)
        expected_circuit.h(0)
        expected_circuit.cx(0, 1)
        expected_circuit.h(1)
        expected_circuit.swap(0, 2)
        expected_circuit.cx(1, 0)
        expected_circuit.h(0)

        routed_dag = insert_swaps_in_circuit(circuit, swaps)
        assert expected_circuit == dag_to_circuit(routed_dag)

    def test_empty_swap(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        swaps = []

        routed_dag = insert_swaps_in_circuit(circuit, swaps)
        assert circuit == dag_to_circuit(routed_dag)
