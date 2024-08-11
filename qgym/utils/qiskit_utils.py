"""This module contains utility functions for interactions with the qiskit library."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
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
    dag = _parse_circuit(circuit)

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


def _parse_circuit(circuit: QuantumCircuit | DAGCircuit) -> DAGCircuit:
    """Create a DAGCircuit from a QuantumCircuit or DAGCircuit."""
    return circuit if isinstance(circuit, DAGCircuit) else circuit_to_dag(circuit)


def _get_qreg_to_int_mapping(
    circuit: QuantumCircuit | DAGCircuit,
) -> dict[Hashable, int]:
    """Create a mapping from the qubits to integer values."""
    return {qubit: circuit.qubits.index(qubit) for qubit in circuit.qubits}


class QiskitMapperWrapper:
    """Wrap any qiskit mapper (:class:`~qiskit.transpiler.Layout`) such that it becomes
    compatible with the qgym framework. This class wraps the qiskit mapper, such that it
    is compatible with the qgym Mapper protocol, which is required for the qgym
    benchmarking tools.
    """

    def __init__(self, qiskit_mapper: Layout) -> None:
        """Init of the :class:`QiskitMapperWrapper`.

        Args:
            qiskit_mapper: The qiskit mapper (:class:`~qiskit.transpiler.Layout`) to
                wrap.
        """
        self.mapper = qiskit_mapper

    def compute_mapping(self, circuit: QuantumCircuit | DAGCircuit) -> NDArray[np.int_]:
        """Compute a mapping of the `circuit` using the provided `qiskit_mapper`.

        Args:
            circuit: Quantum circuit to map.

        Returns:
            Array of which the index represents a physical qubit, and the value a
            virtual qubit.
        """
        dag = _parse_circuit(circuit)

        self.mapper.run(dag)
        layout = self.mapper.property_set["layout"]

        # Convert qiskit layout to qgym mapping
        qreg_to_int = _get_qreg_to_int_mapping(dag)
        iterable = (qreg_to_int[layout[i]] for i in range(dag.num_qubits()))
        return np.fromiter(iterable, int, dag.num_qubits())

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"{self.__class__.__name__}[{self.mapper}]"
