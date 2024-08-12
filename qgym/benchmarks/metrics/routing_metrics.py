"""Metrics to assess the performance of Routers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
from numpy.typing import NDArray


class RoutingSolutionQuality:
    """The :class:`RoutingSolutionQuality` class."""

    def __init__(
        self,
    ) -> None:
        """Init of the :class:`RoutingSolutionQuality` class."""

    def interaction_gates_ratio_loss(
        self,
        initial_interaction_circuit: NDArray[np.int_] | None = None,
        swaps_added: int | None = None,
        final_interaction_circuit: NDArray[np.int_] | None = None,
    ) -> float:
        """Method to calculate the ratio of the final number of interactions divided by
        the initial number of interactions.

        At least two of three need to be specified.

        Args:
            swaps_added: int, represents the number of swaps added during the routing
                process.
            initial_interaction_circuit: An array of 2-tuples of integers, where every
                tuple represents a, not specified, gate acting on the two qubits labeled
                by the integers in the tuples. This is the interaction circuit
                considered before adding the swaps.
            final_interaction_circuit: An array of 2-tuples of integers, where every
                tuple represents a, not specified, gate acting on the two qubits labeled
                by the integers in the tuples. This is the interaction circuit
                considered after adding the swaps.
        """
        if initial_interaction_circuit is not None and swaps_added is not None:
            num_initial_interactions = len(initial_interaction_circuit)
            return (num_initial_interactions + swaps_added) / num_initial_interactions

        if final_interaction_circuit is not None and swaps_added is not None:
            num_final_interactions = len(final_interaction_circuit)
            return num_final_interactions / (num_final_interactions - swaps_added)

        if (
            initial_interaction_circuit is not None
            and final_interaction_circuit is not None
        ):
            num_final_interactions = len(final_interaction_circuit)
            num_initial_interactions = len(initial_interaction_circuit)
            return num_final_interactions / num_initial_interactions

        msg = "at least two of the input variables have to be given"
        raise ValueError(msg)

    def gates_ratio_loss(
        self,
        initial_number_of_gates: int | None = None,
        swaps_added: int | None = None,
        final_number_of_gates: int | None = None,
    ) -> float:
        """Method to calculate the ratio of the final number of gates divided by
        the initial number of gates in a circuit.

        Args:
            initial_number_of_gates: int
            swaps_added: int, represents the number of swaps added during the routing
                process.
            final_number_of_gates: int
        """
        if initial_number_of_gates is not None and swaps_added is not None:
            return (initial_number_of_gates + swaps_added) / initial_number_of_gates

        if final_number_of_gates is not None and swaps_added is not None:
            return final_number_of_gates / (final_number_of_gates - swaps_added)

        if final_number_of_gates is not None and initial_number_of_gates is not None:
            return final_number_of_gates / initial_number_of_gates

        msg = "at least two of the input variables have to be given"
        raise ValueError(msg)

    def average_edge_fidelity_ratio_loss(
        self,
        initial_interaction_circuit: NDArray[np.int_],
        connection_graph: nx.Graph,
        final_interaction_circuit: NDArray[np.int_],
    ) -> float:
        """Method which calculates the average fidelity on the edges, where the fidelity
        of one edge is calculated by consecutively multiplying 1 with the
        connection_graph_edge_fidelity for the used interactions in the interaction
        graph. The averages over all edges before and after adding swaps are calculated
        and divided to get the ratio.

        Args:
            initial_interaction_circuit: An array of 2-tuples of integers, where every
                tuple represents a, not specified, gate acting on the two qubits labeled
                by the integers in the tuples.
            connection_graph: :class:`networkx.Graph` representation of the QPU
                topology. Each node represents a physical qubit and each edge represents
                a connection in the QPU topology.
            final_interaction_circuit: An array of 2-tuples of integers, where every
                tuple represents a, not specified, gate acting on the two qubits labeled
                by the integers in the tuples.
        """

        # TODO: method needs thorough revision, these choices are neither obvious nor
        # correct perse.
        initial_fidelities: dict[tuple[int, int], float] = defaultdict(lambda: 1.0)
        for qubit1, qubit2 in initial_interaction_circuit:
            initial_fidelities[qubit1, qubit2] *= connection_graph[qubit1][qubit2][
                "weight"
            ]
        initial_fidelity_mean = np.mean(list(initial_fidelities.values()))

        final_fidelities: dict[tuple[int, int], float] = defaultdict(lambda: 1.0)
        for qubit1, qubit2 in final_interaction_circuit:
            final_fidelities[qubit1, qubit2] *= connection_graph[qubit1][qubit2][
                "weight"
            ]
        final_fidelity_mean = np.mean(list(final_fidelities.values()))

        return float(initial_fidelity_mean / final_fidelity_mean)

    def average_qubit_fidelity_ratio_loss(
        self,
        initial_interaction_circuit: NDArray[np.int_],
        connection_graph: nx.Graph,
        final_interaction_circuit: NDArray[np.int_],
    ) -> float:
        """Method which calculates the average fidelity on qubits, where the fidelity of
        one qubit is calculated by consecutively multiplying 1 with the
        connection_graph_edge_fidelity for those edge in the interaction graph that are
        connected to that particular qubit. The averages over all qubits before and
        after adding swaps are calculated and divided to get the ratio.

        Args:
            initial_interaction_circuit: An array of 2-tuples of integers, where every
                tuple represents a, not specified, gate acting on the two qubits labeled
                by the integers in the tuples.
            connection_graph: :class:`networkx.Graph` representation of the QPU
                topology. Each node represents a physical qubit and each edge represents
                a connection in the QPU topology.
            final_interaction_circuit: An array of 2-tuples of integers, where every
                tuple represents a, not specified, gate acting on the two qubits labeled
                by the integers in the tuples.
        """
        initial_qubit_fidelity_mean = self._compute_mean_fidelity(
            initial_interaction_circuit, connection_graph
        )
        final_qubit_fidelity_mean = self._compute_mean_fidelity(
            final_interaction_circuit, connection_graph
        )

        return float(initial_qubit_fidelity_mean / final_qubit_fidelity_mean)

    def _compute_mean_fidelity(
        self, interaction_circuit: NDArray[np.int_], connection_graph: nx.Graph
    ) -> float:
        qubit_fidelities: dict[int, float] = defaultdict(lambda: 1.0)
        for qubit1, qubit2 in interaction_circuit:
            fidelity = connection_graph[qubit1][qubit2]["weight"]
            qubit_fidelities[qubit1] *= fidelity
            qubit_fidelities[qubit2] *= fidelity
        return float(np.mean(list(qubit_fidelities.values())))
