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

    def __init__(self, connection_graph: nx.Graph) -> None:
        """Init of the :class:`InitialMappingSolutionQuality` class.

        Args:
            connection_graph: :class:`networkx.Graph` representation of the QPU
                topology. Each node represents a physical qubit and each edge represents
                a connection in the QPU topology.

        """
        self.connection_graph = connection_graph

    def distance_ratio_loss(
        self, interaction_graph: nx.Graph, mapping: NDArray[np.int_]
    ) -> int:
        distance_loss = 0

        for edge in interaction_graph.edges():
            mapped_edge = (mapping[edge[0]], mapping[edge[1]])
            distance_loss += nx.shortest_path_length(
                self.connection_graph, *mapped_edge
            )

        return distance_loss / interaction_graph.number_of_edges()


class AgentPerformance:

    def __init__(
        self,
    ) -> None:
        """Init of the :class:`AgentPerformance` class.

        Args:
        """
