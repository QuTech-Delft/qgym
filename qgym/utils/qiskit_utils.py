"""This module contains utility functions for interactions with the qiskit library."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout


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
        for op_node in dag.two_qubit_ops()
    )

    return interaction_graph


def get_interaction_circuit(circuit: QuantumCircuit | DAGCircuit) -> NDArray[np.int_]:
    """Create and interaction circuit from the provided `circuit`.

    Args:
        circuit: Circuit to produce the interactions circuit. Both
            :class:`~qiskit.circuit.QuantumCircuit` and
            :class:`~qiskit.dagcircuit.DAGCircuit` representations are accepted.

    Returns:
        Interaction circuit of the `circuit`.
    """
    dag = parse_circuit(circuit)

    if dag.multi_qubit_ops():
        msg = "no 3+ qubit operations are supported"
        raise ValueError(msg)

    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        msg = "Interaction circuits are defined for physical circuits only"
        raise ValueError(msg)

    layout = Layout.generate_trivial_layout(dag.qregs["q"])
    interaction_circuit: deque[tuple[int, int]] = deque()

    for op_node in dag.two_qubit_ops():
        qubit1 = layout[op_node.qargs[0]]
        qubit2 = layout[op_node.qargs[1]]
        interaction_circuit.append((qubit1, qubit2))

    if len(interaction_circuit):
        return np.array(interaction_circuit, dtype=np.int_)
    return np.empty((0, 2), dtype=np.int_)


def add_swaps_to_circuit(
    circuit: QuantumCircuit | DAGCircuit, swaps_inserted: Iterable[tuple[int, int, int]]
) -> DAGCircuit:
    dag = parse_circuit(circuit)
    output_dag = dag.copy_empty_like()
    current_layout = Layout.generate_trivial_layout(dag.qregs["q"])
    swaps_iter = iter(swaps_inserted)
    swap_idx, qubit1, qubit2 = next(swaps_iter)

    for layer in dag.serial_layers():
        subdag = layer["graph"]
        for interaction_idx in enumerate(subdag.two_qubit_ops()):
            while interaction_idx == swap_idx:
                # Insert a new layer with the SWAP(s).
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(dag.qregs["q"])

                # create the swap operation
                swap_layer.apply_operation_back(
                    SwapGate(), (qubit1, qubit2), cargs=(), check=False
                )

                # layer insertion
                order = current_layout.reorder_bits(output_dag.qubits)
                output_dag.compose(swap_layer, qubits=order)

                # update current_layout
                current_layout.swap(qubit1, qubit2)

                # get next swap
                try:
                    swap_idx, qubit1, qubit2 = next(swaps_iter)
                except StopIteration:
                    swap_idx = -1

        order = current_layout.reorder_bits(output_dag.qubits)
        output_dag.compose(subdag, qubits=order)

    return output_dag


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
