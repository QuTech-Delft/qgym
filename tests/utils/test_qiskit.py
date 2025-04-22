from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit

from qgym.utils.qiskit import Circuit


class TestParseCircuitLike:
    def test_quantum_circuit(self) -> None:
        circuit = QuantumCircuit(1)
        dag_circuit = Circuit._parse_circuit_like(circuit)
        assert circuit is not dag_circuit
        assert dag_circuit.qregs == {"q": QuantumRegister(1, "q")}
        assert len(list(dag_circuit.nodes())) == 2

    def test_dag_circuit(self) -> None:
        dag_circuit = DAGCircuit()
        assert dag_circuit is Circuit._parse_circuit_like(dag_circuit)

    def test_other_circuit(self) -> None:
        circuit = Circuit(QuantumCircuit(1))
        dag_circuit = Circuit._parse_circuit_like(circuit)
        assert circuit is not dag_circuit
        assert dag_circuit.qregs == {"q": QuantumRegister(1, "q")}
        assert len(list(dag_circuit.nodes())) == 2


class TestGetInteractionGraph:
    @pytest.mark.parametrize(
        ("circuit", "expected_graph"),
        [
            (Circuit(QuantumCircuit(circuit_size)), nx.empty_graph(circuit_size))
            for circuit_size in range(6)
        ],
    )
    def test_empty_circuit(self, circuit: Circuit, expected_graph: nx.Graph) -> None:
        interaction_graph = circuit.get_interaction_graph()
        assert nx.is_isomorphic(interaction_graph, expected_graph)

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_dense_circuit(self, circuit_size: int) -> None:
        qiskit_circuit = QuantumCircuit(circuit_size)
        qiskit_circuit.h(range(circuit_size))
        qiskit_circuit.cx(*zip(*itertools.combinations(range(circuit_size), 2)))
        circuit = Circuit(qiskit_circuit)

        interaction_graph = circuit.get_interaction_graph()
        assert nx.is_isomorphic(interaction_graph, nx.complete_graph(circuit_size))

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_line_circuit(self, circuit_size: int) -> None:
        qiskit_circuit = QuantumCircuit(circuit_size)
        qiskit_circuit.cx(range(circuit_size - 1), range(1, circuit_size))

        circuit = Circuit(qiskit_circuit)
        interaction_graph = circuit.get_interaction_graph()

        assert nx.is_isomorphic(interaction_graph, nx.path_graph(circuit_size))

    def test_multi_qubit_value_error(self) -> None:
        qiskit_circuit = QuantumCircuit(3)
        qiskit_circuit.ccx(0, 1, 2)
        circuit = Circuit(qiskit_circuit)
        with pytest.raises(ValueError, match="3\+ qubit operations are not supported"):
            circuit.get_interaction_graph()


class TestGetInteractionCircuit:
    @pytest.mark.parametrize(
        ("circuit"),
        [Circuit(QuantumCircuit(circuit_size)) for circuit_size in range(1, 6)],
    )
    def test_empty_circuit(self, circuit: Circuit) -> None:
        interaction_circuit = circuit.get_interaction_circuit()
        np.testing.assert_array_equal(interaction_circuit, np.empty((0, 2)))

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_dense_circuit(self, circuit_size: int) -> None:
        qiskit_circuit = QuantumCircuit(circuit_size)
        qiskit_circuit.h(range(circuit_size))
        qiskit_circuit.cx(*zip(*itertools.combinations(range(circuit_size), 2)))
        qiskit_circuit.h(range(circuit_size))
        circuit = Circuit(qiskit_circuit)

        expected_result = np.array([*itertools.combinations(range(circuit_size), 2)])
        interaction_circuit = circuit.get_interaction_circuit()
        np.testing.assert_array_equal(interaction_circuit, expected_result)

    @pytest.mark.parametrize("circuit_size", range(2, 6))
    def test_line_circuit(self, circuit_size: int) -> None:
        qiskit_circuit = QuantumCircuit(circuit_size)
        qiskit_circuit.h(range(circuit_size))
        qiskit_circuit.cx(range(circuit_size - 1), range(1, circuit_size))
        qiskit_circuit.h(range(circuit_size))

        circuit = Circuit(qiskit_circuit)

        expected_result = np.array(
            [np.arange(circuit_size - 1), np.arange(1, circuit_size)]
        ).T
        interaction_circuit = circuit.get_interaction_circuit()
        np.testing.assert_array_equal(interaction_circuit, expected_result)

    def test_multi_qubit_value_error(self) -> None:
        qiskit_circuit = QuantumCircuit(3)
        qiskit_circuit.ccx(0, 1, 2)
        circuit = Circuit(qiskit_circuit)
        with pytest.raises(ValueError, match="3\+ qubit operations are not supported"):
            circuit.get_interaction_circuit()

    @pytest.mark.parametrize(
        "circuit",
        [
            Circuit(QuantumCircuit()),
            Circuit(QuantumCircuit(QuantumRegister(1, "q"), QuantumRegister(1, "a"))),
            Circuit(QuantumCircuit(QuantumRegister(1, "a"))),
        ],
        ids=["no registers", "multiple registers", "wrong name"],
    )
    def test_non_physical_circuit_error(self, circuit: Circuit) -> None:
        expected_msg = "Interaction circuits are defined for physical circuits only"
        with pytest.raises(ValueError, match=expected_msg):
            circuit.get_interaction_circuit()


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
        qiskit_circuit = QuantumCircuit(5)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.cx(1, 2)
        qiskit_circuit.cx(2, 3)
        qiskit_circuit.cx(3, 4)
        circuit = Circuit(qiskit_circuit)

        swaps = [(1, 3, 4), (1, 2, 3), (3, 1, 2)]

        expected_circuit = QuantumCircuit(5)
        expected_circuit.cx(0, 1)
        expected_circuit.swap(3, 4)
        expected_circuit.swap(2, 3)
        expected_circuit.cx(1, 3)
        expected_circuit.swap(1, 2)
        expected_circuit.cx(3, 4)
        expected_circuit.cx(4, 1)

        routed_circuit = circuit.insert_swaps_in_circuit(swaps)
        assert expected_circuit == routed_circuit.get_qiskit_quantum_circuit()

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
        qiskit_circuit = QuantumCircuit(3)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        qiskit_circuit.h(1)
        qiskit_circuit.cx(1, 2)
        qiskit_circuit.h(2)
        circuit = Circuit(qiskit_circuit)

        swaps = [(1, 0, 2)]

        expected_circuit = QuantumCircuit(3)
        expected_circuit.h(0)
        expected_circuit.cx(0, 1)
        expected_circuit.h(1)
        expected_circuit.swap(0, 2)
        expected_circuit.cx(1, 0)
        expected_circuit.h(0)

        routed_circuit = circuit.insert_swaps_in_circuit(swaps)
        assert expected_circuit == routed_circuit.get_qiskit_quantum_circuit()

    def test_empty_swap(self) -> None:
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.h(0)
        qiskit_circuit.cx(0, 1)
        circuit = Circuit(qiskit_circuit)

        swaps = []

        routed_circuit = circuit.insert_swaps_in_circuit(swaps)
        assert qiskit_circuit == routed_circuit.get_qiskit_quantum_circuit()
