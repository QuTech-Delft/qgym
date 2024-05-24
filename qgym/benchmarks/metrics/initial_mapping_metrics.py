"""Metrics to assess the performance of InitialMappers.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

import networkx as nx
import numpy as np
from numpy.typing import NDArray


class InitialMappingSolutionQuality:

    def __init__(
        self,
        connection_graph: nx.Graph,
    ) -> None:
        # pylint: disable=line-too-long
        """Init of the :class:`~qgym.benchmarks.metrics.initial_mapping_metrics.InitialMappingSolutionQuality` class.

        Args:
            connection_graph: `networkx Graph <https://networkx.org/documentation/stable/reference/classes/graph.html>`_
                representation of the QPU topology. Each node represents a physical
                qubit and each edge represents a connection in the QPU topology.
        """
        self.connection_graph = connection_graph

    def distance_ratio_loss(
        self,
        interaction_graph: nx.Graph,
        mapping: NDArray[np.int_],
    ) -> int:
        distance_loss = 0

        for edge in interaction_graph.edges():
            mapped_edge = (mapping[edge[0]], mapping[edge[1]])
            if mapped_edge not in self.connection_graph.edges():
                distance_loss += nx.shortest_path_length(
                    self.connection_graph(),
                    source=mapped_edge[0],
                    target=mapped_edge[1],
                )

        total_routing_distance = interaction_graph.number_of_edges() + distance_loss
        return total_routing_distance / interaction_graph.number_of_edges()


class AgentPerformance:

    def __init__(
        self,
    ) -> None:
        # pylint: disable=line-too-long
        """Init of the :class:`~qgym.benchmarks.metrics.initial_mapping_metrics.AgentPerformance` class.

        Args:
        """
