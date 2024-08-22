"""This module contains utility functions for interactions with the qiskit library."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit


def get_interaction_graph(circuit: QuantumCircuit | DAGCircuit) -> nx.Graph:
    """Create and interaction graph from the provided `circuit`.

    Args:
        circuit: Circuit to produce the interactions from. Both
            :class:`~qiskit.circuit.QuantumCircuit` and
            :class:`~qiskit.dagcircuit.DAGCircuit` representations are accepted.

    Returns:
        Interaction graph of the `circuit`.
    """
    dag = parse_circuit(circuit)

    if dag.multi_qubit_ops():
        msg = "no 3+ qubit operations are supported"
        raise ValueError(msg)

    qreg_to_int = _get_qreg_to_int_mapping(dag)
    interaction_graph: nx.Graph = nx.empty_graph(dag.num_qubits())

    interaction_graph.add_edges_from(
        (qreg_to_int[op_node.qargs[0]], qreg_to_int[op_node.qargs[1]])
        for op_node in dag.op_nodes(include_directives=False)
        if len(op_node.qargs) == 2
    )

    return interaction_graph


def parse_circuit(circuit: QuantumCircuit | DAGCircuit) -> DAGCircuit:
    """Create a DAGCircuit from a QuantumCircuit or DAGCircuit.:

    Args:
        circuit: Circuit to parse.

    Returns:
        DAGCircuit representation of the circuit.
    """
    return circuit if isinstance(circuit, DAGCircuit) else circuit_to_dag(circuit)


def _get_qreg_to_int_mapping(
    circuit: QuantumCircuit | DAGCircuit,
) -> dict[Hashable, int]:
    """Create a mapping from the qubits to integer values."""
    return {qubit: circuit.qubits.index(qubit) for qubit in circuit.qubits}
