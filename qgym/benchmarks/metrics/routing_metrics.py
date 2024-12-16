"""Metrics to assess the performance of Routers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from abc import abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from numpy.typing import ArrayLike
from qgym.utils.qiskit_utils import parse_circuit
from qgym.utils.input_validation import check_int
from qgym.utils.input_parsing import parse_connection_graph, has_fidelity
from typing import SupportsInt
from qiskit.transpiler import Layout


@runtime_checkable
class RoutingMetric(Protocol):
    """Protocol that a metric for qubit routing should follow."""

    @abstractmethod
    def compute(
        self,
        input_circuit: QuantumCircuit | DAGCircuit,
        routed_circuit: QuantumCircuit | DAGCircuit,
    ) -> float:
        """Compute the metric for the provided `input_circuit` and `routed_circuit`."""


class InteractionRatioLoss(RoutingMetric):
    """The :class:`InteractionRatioLoss` class."""

    def __init__(self, swap_penalty: SupportsInt = 3) -> None:
        """Init of :class:`InteractionRatioLoss`.

        Args:
            swap_penalty: Number of gates to use to decompose the SWAP gate. Since a
                SWAP gate is often decomposed using 3 CNOT gates, the default is 3.
        """
        self.swap_penalty = check_int(swap_penalty, "swap_penalty", l_bound=1)

    def compute(
        self,
        input_circuit: QuantumCircuit | DAGCircuit,
        routed_circuit: QuantumCircuit | DAGCircuit,
    ) -> float:
        """Method to calculate the ratio of the input and output circuit.

        The ratio loss is defined by the number of 2 qubit gates in the `output_circuit`
        devided by the number of 2 qubit gates in the `input_circut`. A score of 1 is
        thus a perfect score and the lowest value that can be returned.

        Args:
            input_circuit: Input circuit before routing was performed.
            routed_circut: Routed version of the input circuit.

        Returns:
            The routing solution quality ratio.
        """
        input_num_iteractions = self.get_num_interactions(input_circuit)
        routed_num_iteractions = self.get_num_interactions(routed_circuit)
        return routed_num_iteractions / input_num_iteractions

    def get_num_interactions(self, circuit: QuantumCircuit | DAGCircuit) -> int:
        """Get the number of interactions from the `circuit`.

        Args:
            circuit: Circuit to count the number of interactions from.
            swap_penalty: Number of gates to use to decompose the SWAP gate. Since a
                SWAP gate is often decomposed using 3 CNOT gates, the default is 3.

        Returns:
            Number of 2 qubit interactions, where each SWAP gate is counted
            `swap_penalty` times.
        """
        dag = parse_circuit(circuit)
        return sum(
            self.swap_penalty if gate.name == "swap" else 1
            for gate in dag.two_qubit_ops()
        )


class MinEdgeFidelityRatioLoss(RoutingMetric):
    """The :class:`InteractionRatioLoss` class."""

    def __init__(
        self, connection_graph: nx.Graph, swap_penalty: SupportsInt = 3
    ) -> None:
        """Init of :class:`InteractionRatioLoss`.

        Args:
            connection_graph: :class:`networkx.Graph` representation of the QPU
                topology. Each node represents a physical qubit and each edge represents
                a connection in the QPU topology.
            swap_penalty: Number of gates to use to decompose the SWAP gate. Since a
                SWAP gate is often decomposed using 3 CNOT gates, the default is 3.
        """
        self.connection_graph = parse_connection_graph(connection_graph)
        if not has_fidelity(self.connection_graph):
            msg = "connection_graph does not contain fidelity"
            raise ValueError(msg)
        self.swap_penalty = check_int(swap_penalty, "swap_penalty", l_bound=1)

    def get_n_interactions_per_connection(
        self, circuit: QuantumCircuit | DAGCircuit
    ) -> defaultdict[frozenset[int], int]:
        dag = parse_circuit(circuit)
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            msg = "minimum fidelity ratio can only be computed for physical circuits"
            raise ValueError(msg)

        layout = Layout.generate_trivial_layout(dag.qregs["q"])
        n_interactions = defaultdict(int)
        for gate in dag.two_qubit_ops():
            qubit1 = layout[gate.qargs[0]]
            qubit2 = layout[gate.qargs[1]]
            count = self.swap_penalty if gate.name == "swap" else 1
            n_interactions[frozenset(qubit1, qubit2)] += count

        return n_interactions

    def compute(
        self,
        input_circuit: QuantumCircuit | DAGCircuit,
        routed_circuit: QuantumCircuit | DAGCircuit,
    ) -> float:
        """Method to calculate the ratio of the input and output circuit.

        The minimum edge fidelity is computed as $\min(\{f_e^{n_e}| e \in E\})$. Here
        $E$ is the edge set of the connection graph $G=(V,E)$, `f(e)` is the fidelity of
        edge $e$ and $n_e$ is the number of times edge $e$ is used in the quantum
        circuit. The minimum edge fidelity routing solution quality ratio that is
        returned is the minimum edge fidelity of the input circuit divided by the
        minimum edge fidelity of the routed circuit.

        Args:
            input_circuit: Input circuit before routing was performed.
            routed_circut: Routed version of the input circuit.

        Returns:
            The minimum edge fidelity routing solution quality ratio.
        """
        n_interactions_input = self.get_n_interactions_per_connection(input_circuit)
        n_interactions_routed = self.get_n_interactions_per_connection(routed_circuit)

        min_fidelity_input = min(
            pow(self.connection_graph[qubit1][qubit2]["weight"], n_interactions)
            for (qubit1, qubit2), n_interactions in n_interactions_input.items()
        )
        min_fidelity_routed = min(
            pow(self.connection_graph[qubit1][qubit2]["weight"], n_interactions)
            for (qubit1, qubit2), n_interactions in n_interactions_routed.items()
        )

        return float(min_fidelity_input / min_fidelity_routed)
