"""This module contains generators for qiskit circuits."""

from __future__ import annotations

from typing import Iterator, SupportsFloat, SupportsInt

import networkx as nx
import numpy as np
from numpy.random import Generator
from qiskit import QuantumCircuit

from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_int, check_real


class MaxCutQAOAGenerator(Iterator[QuantumCircuit]):
    """Generate simple QAOA circuits for random MaxCut problems."""

    def __init__(
        self,
        n_nodes: SupportsInt,
        edge_probability: SupportsFloat,
        n_layers: SupportsInt = 1,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`MaxCutQAOAGenerator`.

        The QAOA circuit is made for the MaxCut problem defined on a randomly generated
        Erdős-Rényi graph of size `n_nodes` with an edge probability of
        `edge_probability`. The QAOA circuit is a simple circuit which initializes all
        qubits in the |+⟩ state. Next repeated cost and mixer layers are added
        `n_layers` times. The mixer layer consists of parameterized Rx gates on all
        qubits. The parameters for the circuit are randomly generated values in
        $[0, pi)$.

        Args:
            n_nodes: Number of nodes in the generated graphs for the underlying MaxCut
                problem. This is equal to the number of qubits the output circuit will
                have.
            edge_probability: Probability of an edge appearing in the generated graphs
                for the underlying MaxCut problem.
            n_layers: Number of times the cost and mixer layers should be repeated.
            seed: Seed to use for the random graph and parameter generation.

        Returns:
            Simple QAOA circuit for the MaxCut problem on the provided `graph` of depth
            `self.p` with random parameters.
        """
        self.n_nodes = check_int(n_nodes, "n_nodes", l_bound=1)
        self.edge_probability = check_real(
            edge_probability, "edge_probability", l_bound=0, u_bound=1
        )
        self.n_layers = check_int(n_layers, "n_layers", l_bound=1)
        self.rng = parse_seed(seed)
        self.finite = False

    def __next__(self) -> QuantumCircuit:
        """Simple QAOA circuit for the MaxCut problem of a randomly generated graph."""
        graph = nx.fast_gnp_random_graph(self.n_nodes, self.edge_probability, self.rng)
        return self._maxcut_qaoa_circuit(graph)

    def _maxcut_qaoa_circuit(self, graph: nx.Graph) -> QuantumCircuit:
        """Create a simple QAOA circuit for the MaxCut problem on the provided `graph`.

        The parameters for the circuit are randomly generated values in $[0, pi)$
        and the number of repeated layers is `self.p`.

        Args:
            graph: Graph to create the MaxCut QAOA circuit for.

        Returns:
            Simple QAOA circuit for the MaxCut problem on the provided `graph` of depth
            `self.p` with random parameters.
        """
        circuit = QuantumCircuit(self.n_nodes)
        circuit.h(range(self.n_nodes))
        beta, gamma = self.rng.uniform(low=0, high=np.pi, size=(2, self.n_layers))
        for beta_i, gamma_i in zip(beta, gamma):
            for node1, node2 in graph.edges():
                circuit.rzz(gamma_i, node1, node2)
            circuit.rx(beta_i, range(self.n_nodes))
        circuit.measure_all()
        return circuit

    def __repr__(self) -> str:
        """String representation of :class:`MaxCutQAOAGenerator`."""
        return (
            f"{self.__class__.__name__}[n_nodes={self.n_nodes}, edge_probability="
            f"{self.edge_probability}, n_layers={self.n_layers}, rng={self.rng}]"
        )
