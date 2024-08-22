from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import Layout

from qgym.utils.qiskit_utils import _get_qreg_to_int_mapping, parse_circuit


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
        dag = parse_circuit(circuit)

        self.mapper.run(dag)
        layout = self.mapper.property_set["layout"]

        # Convert qiskit layout to qgym mapping
        qreg_to_int = _get_qreg_to_int_mapping(dag)
        iterable = (qreg_to_int[layout[i]] for i in range(dag.num_qubits()))
        return np.fromiter(iterable, int, dag.num_qubits())

    def __repr__(self) -> str:
        """String representation of the wrapper."""
        return f"{self.__class__.__name__}[{self.mapper}]"
