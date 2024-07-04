from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit


def get_ineraction_graph(circuit: QuantumCircuit | DAGCircuit) -> nx.Graph:
    """Create and interaction graph from the provided `circuit`.
    
    Args:
        circuit: Circuit to produce the interactions from. Both ``QuantumCircuit`` and
            ``DAGCircuit`` representations are accepted.
    
    Returns:
        Interaction graph of the `circuit`.
    """
    dag = _parse_circuit(circuit)

    if dag.multi_qubit_ops():
        raise ValueError("No 3+ qubit operations are supported.")

    qreg_to_int = _get_qreg_to_int_mapping(dag)
    interaction_graph: nx.Graph = nx.empty_graph(dag.num_qubits())

    for op_node in dag.op_nodes(include_directives=False):
        if len(op_node.qargs) == 1:
            continue
        qubit1, qubit2 = map(qreg_to_int, op_node.qargs)
        interaction_graph.add_edge(qubit1, qubit2)

    return interaction_graph

def _parse_circuit(circuit: QuantumCircuit | DAGCircuit) -> DAGCircuit:
    return circuit if isinstance(circuit, DAGCircuit) else circuit_to_dag(circuit)   

def _get_qreg_to_int_mapping(circuit: QuantumCircuit | DAGCircuit) -> dict[Hashable, int]:
    return  {qubit: circuit.qubits.index(qubit) for qubit in circuit.qubits}

class QiskitMapperWrapper:

    def __init__(self, qiskit_mapper) -> None:
        self.mapper = qiskit_mapper
    
    def compute_mapping(self, circuit: QuantumCircuit | DAGCircuit) -> NDArray[np.int_]:
        dag = _parse_circuit(circuit)

        self.mapper.run(dag)
        layout = self.mapper.property_set["layout"]

        # Convert qiskit layout to qgym mapping
        qreg_to_int = _get_qreg_to_int_mapping(dag)
        iterable = (qreg_to_int[layout[i]] for i in range(dag.num_qubits()))
        return np.fromiter(iterable, int, dag.num_qubits())


if __name__ == "__main__":
    from mqt.bench import get_benchmark
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.passes import VF2Layout

    coupling_map = CouplingMap([[0,1], [1,2], [2,3], [3,4], [4,5], [5,0]])
    circuit = get_benchmark("graphstate", "nativegates", 6)

    qiskit_mapper = VF2Layout(coupling_map)
    mapper = QiskitMapperWrapper(qiskit_mapper)
    mapping = mapper.compute_mapping(circuit)

    print(mapping)


    