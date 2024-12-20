"""This method contains utility functions for qiskit object."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


class Circuit:
    """Wrapper for Qiskit Quantum Circuits.

    Contains utility functions helpful for interacting with qgym.
    """

    def __init__(self, circuit: CircuitLike) -> None:
        """Init of :class:`Circuit`.

        Args:
            circuit: CircuitLike argument to wrap.
        """
        self.dag = self._parse_circuit_like(circuit)

    def get_qiskit_quantum_circuit(self) -> QuantumCircuit:
        """Convert the circuit to a qiskit ``QuantumCircuit``."""
        return dag_to_circuit(self.dag)

    @staticmethod
    def _parse_circuit_like(circuit: CircuitLike) -> DAGCircuit:
        if isinstance(circuit, Circuit):
            return circuit.dag
        if isinstance(circuit, QuantumCircuit):
            return circuit_to_dag(circuit)
        return circuit

    def get_interaction_graph(self) -> nx.Graph:
        """Create and interaction graph of the circuit.

        Returns:
            Interaction graph of the circuit.
        """
        if self.dag.multi_qubit_ops():
            msg = "no 3+ qubit operations are supported"
            raise ValueError(msg)

        layout = Layout.generate_trivial_layout(*self.dag.qregs.values())
        interaction_graph: nx.Graph = nx.empty_graph(self.dag.num_qubits())

        interaction_graph.add_edges_from(
            tuple(map(layout.__getitem__, gate.qargs))
            for gate in self.dag.two_qubit_ops()
        )

        return interaction_graph

    def get_interaction_circuit(self) -> NDArray[np.int_]:
        """Create and interaction circuit from the circuit.

        Returns:
            Interaction circuit representation of the circuit.
        """
        if self.dag.multi_qubit_ops():
            msg = "no 3+ qubit operations are supported"
            raise ValueError(msg)

        if len(self.dag.qregs) != 1 or self.dag.qregs.get("q", None) is None:
            msg = "Interaction circuits are defined for physical circuits only"
            raise ValueError(msg)

        layout = Layout.generate_trivial_layout(self.dag.qregs["q"])
        interaction_circuit: deque[tuple[int, int]] = deque()

        for op_node in self.dag.two_qubit_ops():
            qubit1 = layout[op_node.qargs[0]]
            qubit2 = layout[op_node.qargs[1]]
            interaction_circuit.append((qubit1, qubit2))

        if len(interaction_circuit):
            return np.array(interaction_circuit, dtype=np.int_)
        return np.empty((0, 2), dtype=np.int_)

    def insert_swaps_in_circuit(
        self, swaps_inserted: Iterable[tuple[int, int, int]]
    ) -> Circuit:
        """Insert the provided swap gates in the quantum circuit.

        Args:
            swaps_inserted: Swap gated to insert. Iterable of tuples (g_idx, q1, q2).
                Each tuple represents a swap gate, where g_idx is the index of the two
                qubit gate before which the swap gate needs to be inserted. The swap is
                performed on q1 and q2.

        Returns:
            A :class:`Circuit` with the provided swap gates inserted.
        """
        output_dag = self.dag.copy_empty_like()
        current_layout = Layout.generate_trivial_layout(self.dag.qregs["q"])
        swaps_iter = iter(swaps_inserted)
        try:
            swap_idx, qubit1, qubit2 = next(swaps_iter)
        except StopIteration:
            # No swaps to insert
            return deepcopy(self)
        interaction_idx = 0

        for layer in self.dag.serial_layers():
            subdag = layer["graph"]
            for _ in subdag.two_qubit_ops():
                while interaction_idx == swap_idx:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(self.dag.qregs["q"])

                    # create the swap operation
                    swap_layer.apply_operation_back(
                        SwapGate(),
                        (current_layout[qubit1], current_layout[qubit2]),
                        cargs=(),
                        check=False,
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

                interaction_idx += 1

            order = current_layout.reorder_bits(output_dag.qubits)
            output_dag.compose(subdag, qubits=order)

        return Circuit(output_dag)


CircuitLike = Union[Circuit, QuantumCircuit, DAGCircuit]
