"""
This module contains a class used for rendering the initial mapping environment.
"""

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pygame
from networkx import Graph
from numpy.typing import NDArray
from pygame import gfxdraw

# Define some colors used during rendering
WHITE = (225, 225, 225)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
RED = (225, 0, 0)
GREEN = (0, 225, 0)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 0, 225)


class InitialMappingVisualiser:
    """
    Visualiser class for the initial mapping environment
    """

    def __init__(self, connection_graph: Graph) -> None:

        self.connection_graph = connection_graph
        self.connection_graph_nodes = connection_graph.nodes
        self.connection_graph_matrix = nx.to_scipy_sparse_array(connection_graph)

        # Rendering data
        self.screen = None
        self.is_open = False
        self.screen_width = 1300
        self.screen_height = 730

        # initialize rectangles
        self.subscreen1 = None
        self.subscreen2 = None
        self.subscreen3 = None
        self.init_subscreen_rectangles()

    def init_subscreen_rectangles(self, padding: int = 10) -> None:
        """
        Initialize the pygame `Rect` objects used for drawing the subscreens.

        :param padding: The padding to be used inbetween the subscreens.
        """

        small_screen_width = self.screen_width / 2 - 1.5 * padding
        small_screen_height = self.screen_height / 2 - 1.5 * padding

        large_screen_width = self.screen_width / 2 - 1.5 * padding
        large_screen_height = self.screen_height - 2 * padding

        screen1_pos = (padding, padding)
        screen2_pos = (padding, small_screen_height + 2 * padding)
        screen3_pos = (small_screen_width + 2 * padding, padding)

        self.subscreen1 = pygame.Rect(
            screen1_pos[0], screen1_pos[1], small_screen_width, small_screen_height
        )
        self.subscreen2 = pygame.Rect(
            screen2_pos[0], screen2_pos[1], small_screen_width, small_screen_height
        )
        self.subscreen3 = pygame.Rect(
            screen3_pos[0], screen3_pos[1], large_screen_width, large_screen_height
        )

    def render(
        self, state: Dict[str, Any], interaction_graph: nx.Graph, mode: str
    ) -> Any:
        """
        Render the current state using pygame. The upper left screen shows the
        connection graph. The lower left screen the interaction graph. The
        right screen shows the mapped graph. Gray edges are unused, green edges
        are mapped correctly and red edges need at least on swap.

        :param state: state to render
        :param interaction_graph: interaction graph to render
        :param mode: Mode to start pygame for ("human" and "rgb_array" are supported).
        :raise ValueError: When an invalid mode is provided.
        :return: In 'human' mode returns a boolean value encoding whether the pygame screen is open. In `rgb_array` mode
            returns an RGB array encoding of the rendered image.
        """

        if self.screen is None:
            self.start(mode)

        pygame.time.delay(10)

        self.screen.fill(GRAY)

        pygame.draw.rect(self.screen, WHITE, self.subscreen1)
        pygame.draw.rect(self.screen, WHITE, self.subscreen2)
        pygame.draw.rect(self.screen, WHITE, self.subscreen3)

        mapped_graph = self._get_mapped_graph(state)

        self._draw_graph(self.connection_graph, self.subscreen1)
        self._draw_graph(interaction_graph, self.subscreen2)
        self._draw_graph(
            mapped_graph, self.subscreen3, pivot_graph=self.connection_graph
        )

        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            return self.is_open
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            raise ValueError(
                f"You provided an invalid mode '{mode}',"
                f" the only supported modes are 'human' and 'rgb_array'."
            )

    def _get_mapped_graph(self, state: Dict[str, Any]) -> nx.Graph:
        """
        Constructs a mapped graph. In this graph gray edges are unused, green
        edges are mapped correctly and red edges need at least on swap. This
        function is used during rendering.

        :param state: state to render
        :return: Mapped graph
        """

        mapping = state["mapping_dict"]

        # Make the adjacency matrix of the mapped graph
        mapped_adjacency_matrix = np.zeros(self.connection_graph_matrix.shape)
        for map_i, i in mapping.items():
            for map_j, j in mapping.items():
                mapped_adjacency_matrix[i, j] = state["interaction_graph_matrix"][
                    map_i, map_j
                ]

        # Make a networkx graph of the mapped graph
        graph = nx.Graph()
        for i in range(self.connection_graph_matrix.shape[0]):
            graph.add_node(i)

        for i in range(self.connection_graph_matrix.shape[0]):
            for j in range(self.connection_graph_matrix.shape[1]):
                self._add_colored_edge(graph, mapped_adjacency_matrix, (i, j))

        # Relabel nodes for drawing
        nodes_mapping = dict(
            [(i, node) for i, node in enumerate(self.connection_graph_nodes)]
        )
        graph = nx.relabel_nodes(graph, nodes_mapping)

        return graph

    def _add_colored_edge(
        self,
        graph: nx.Graph,
        mapped_adjacency_matrix: NDArray,
        edge: Tuple[int, int],
    ) -> None:
        """
        Utility function for making the mapped graph. Gives and edge of the
        graph a certain color. Gray edges are unused, green edges are mapped
        correctly and red edges need at least on swap.

        :param graph: The graph of which the edges must be colored.
        :param mapped_adjacency_matrix: the adjacency matrix of the mapped graph.
        :param edge: The edge to color.
        """

        (i, j) = edge
        is_connected = self.connection_graph_matrix[i, j] != 0
        is_mapped = mapped_adjacency_matrix[i, j] != 0
        if not is_connected and is_mapped:
            graph.add_edge(i, j, color="red")
        if is_connected and is_mapped:
            graph.add_edge(i, j, color="green")
        if is_connected and not is_mapped:
            graph.add_edge(i, j, color="gray")

    def _draw_graph(
        self,
        graph: nx.Graph,
        subscreen: pygame.Rect,
        pivot_graph: Optional[nx.Graph] = None,
    ) -> None:
        """
        Draws a graph on one of the subscreens.

        :param graph: the graph to be drawn.
        :param subscreen: the subscreen on which the graph must be drawn.
        :param pivot_graph: optional graph for which the spectral structure
            will be used for visualisation.
        """

        if pivot_graph is None:
            node_positions = self._get_render_positions(graph, subscreen)
        else:
            assert graph.nodes == pivot_graph.nodes
            node_positions = self._get_render_positions(pivot_graph, subscreen)

        # Draw the nodes of the graph on the subscreen
        for node, pos in node_positions.items():
            gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 5, BLUE)
        # Draw the edges of the graph on the subscreen
        for (u, v) in graph.edges():
            pos_u = node_positions[u]
            pos_v = node_positions[v]
            color = BLACK
            if "color" in graph.edges[u, v]:
                if graph.edges[u, v]["color"] == "red":
                    color = RED
                if graph.edges[u, v]["color"] == "green":
                    color = GREEN
                if graph.edges[u, v]["color"] == "gray":
                    color = DARK_GRAY
            pygame.draw.aaline(self.screen, color, pos_u, pos_v)

    @staticmethod
    def _get_render_positions(
        graph: nx.Graph, subscreen: pygame.Rect
    ) -> Dict[Any, Tuple[float, float]]:
        """
        Utility function used during render. Give the positions of the nodes
        of a graph on a given subscreen.

        :param graph: the graph of which the node positions must be determined.
        :param subscreen: the subscreen on which the graph will be drawn.
        :return: a dictionary where the keys are the names of the nodes, and the
            values are the coordinates of these nodes.
        """

        x_scaling = 0.45 * subscreen.width
        y_scaling = 0.45 * subscreen.height
        x_offset = subscreen.centerx
        y_offset = subscreen.centery
        node_positions = nx.spectral_layout(graph)
        for node in node_positions:
            node_positions[node][0] += node_positions[node][0] * x_scaling + x_offset
            node_positions[node][1] += node_positions[node][1] * y_scaling + y_offset
        return node_positions

    def start(self, mode: str) -> None:
        """
        Start pygame in the given mode.

        :param mode: Mode to start pygame for ("human" and "rgb_array" are supported).
        :raise ValueError: When an invalid mode is provided.
        """

        pygame.display.init()

        if mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        elif mode == "rgb_array":
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            raise ValueError(
                f"You provided an invalid mode '{mode}',"
                f" the only supported modes are 'human' and 'rgb_array'."
            )

        pygame.display.set_caption("Mapping Environment")
        self.is_open = True

    def close(self) -> None:
        """
        Closed the screen used for rendering.
        """

        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            self.screen = None
