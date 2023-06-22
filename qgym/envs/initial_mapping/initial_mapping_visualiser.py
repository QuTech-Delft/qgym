"""This module contains a class used for rendering a ``InitialMapping`` environment."""
from typing import Any, Dict, Tuple, Union

import networkx as nx
import numpy as np
import pygame
from networkx import Graph
from numpy.typing import NDArray
from pygame import gfxdraw

from qgym.envs.initial_mapping.initial_mapping_state import InitialMappingState
from qgym.templates.visualiser import Visualiser

# Define some colors used during rendering
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
RED = (189, 18, 33)
GREEN = (174, 168, 0)
BLUE = (113, 164, 195)

# pylint: disable=invalid-name


class InitialMappingVisualiser(Visualiser):
    """Visualiser class for the ``InitialMapping`` environment."""

    def __init__(self, connection_graph: Graph) -> None:
        """Init of the ``InitialMappingVisualiser``.

        :param connection_graph: ``networkx.Graph`` representation of the connection
            graph.
        """
        # Rendering data
        self.screen_dimensions = (1300, 730)
        self.font_size = 30

        self.screen = None
        self.subscreens = self._init_subscreen_rectangles()
        self.font: Dict[str, pygame.font.Font] = {}

        self.colors = {
            "nodes": BLUE,
            "basic_edge": BLACK,
            "unused_edge": GRAY,
            "used_edge": GREEN,
            "missing_edge": RED,
            "text": BLACK,
            "background": WHITE,
        }

        # Save evrything we need to know about the graphs
        self.graphs = {
            "connection": {
                "render_positions": self._get_render_positions(
                    connection_graph, self.subscreens[0]
                ),
                "nodes": connection_graph.nodes,
                "edges": connection_graph.edges,
                "matrix": nx.to_numpy_array(connection_graph),
            },
            "interaction": {"render_positions": {}},
            "mapped": {
                "render_positions": self._get_render_positions(
                    connection_graph, self.subscreens[2]
                ),
            },
        }

    def _init_subscreen_rectangles(
        self, padding: int = 20
    ) -> Tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        """Initialize the ``pygame.Rect`` objects used for drawing the subscreens.

        :param padding: The padding to be used inbetween the subscreens.
        """
        small_screen_width = self.screen_width / 2 - 0.5 * padding
        small_screen_height = self.screen_height / 2 - self.header_spacing
        small_screen_shape = (small_screen_width, small_screen_height)

        large_screen_width = self.screen_width / 2 - 0.5 * padding
        large_screen_height = self.screen_height - self.header_spacing
        large_screen_shape = (large_screen_width, large_screen_height)

        screen1_pos = (0, self.header_spacing)
        screen2_pos = (0, self.header_spacing + self.screen_height / 2)
        screen3_pos = (small_screen_width + padding, self.header_spacing)

        subscreen1 = pygame.Rect(screen1_pos, small_screen_shape)
        subscreen2 = pygame.Rect(screen2_pos, small_screen_shape)
        subscreen3 = pygame.Rect(screen3_pos, large_screen_shape)
        return subscreen1, subscreen2, subscreen3

    def render(
        self, state: InitialMappingState, mode: str
    ) -> Union[bool, NDArray[np.int_]]:
        """Render the current state using ``pygame``. The upper left screen shows the
        connection graph. The lower left screen the interaction graph. The right screen
        shows the mapped graph. Gray edges are unused, green edges are mapped correctly
        and red edges need at least on swap.

        :param state: State to render.
        :param mode: Mode to start pygame for ("human" and "rgb_array" are supported).
        :raise ValueError: When an invalid mode is provided.
        :return: In 'human' mode returns a boolean value encoding whether the ``pygame``
            screen is open. In 'rgb_array' mode returns an RGB array encoding of the
            rendered image.
        """
        if self.screen is None:
            self.screen = self._start_screen("Mapping Environment", mode)
            pygame.font.init()

        if len(self.font) == 0:
            self.font["header"] = pygame.font.SysFont("Arial", self.font_size)

        pygame.time.delay(10)

        self.screen.fill(self.colors["background"])

        mapped_graph = self._get_mapped_graph(
            state.mapping_dict, state.graphs["interaction"]["matrix"]
        )
        self._draw_connection_graph(self.screen)
        self._draw_interaction_graph(
            self.screen, state.steps_done, state.graphs["interaction"]["graph"]
        )
        self._draw_mapped_graph(self.screen, mapped_graph)

        self._draw_header("Connection Graph", self.subscreens[0], self.screen)
        self._draw_header("Interaction Graph", self.subscreens[1], self.screen)
        self._draw_header("Mapped Graph", self.subscreens[2], self.screen)

        return self._display(mode)

    def _get_mapped_graph(
        self, mapping: Dict[int, int], interaction_graph_matrix: NDArray[np.float_]
    ) -> nx.Graph:
        """Construct a mapped graph. In this graph gray edges are unused, green edges
        are mapped correctly and red edges need at least on swap. This function is used
        during rendering.

        :param mapping: Mapping dictionary of the state to render.
        :interaction_graph_matrix: Interaction graph matrix of the current interaction
            graph.
        :return: Mapped graph.
        """
        # Make the adjacency matrix of the mapped graph
        n_nodes = len(self.graphs["connection"]["nodes"])
        mapped_matrix = np.zeros_like(self.graphs["connection"]["matrix"])
        for map_i, i in mapping.items():
            for map_j, j in mapping.items():
                mapped_matrix[i, j] = interaction_graph_matrix[n_nodes * map_i + map_j]

        # Make a networkx graph of the mapped graph
        graph = nx.Graph()
        for i in range(mapped_matrix.shape[0]):
            graph.add_node(i)

        for i in range(mapped_matrix.shape[0]):
            for j in range(mapped_matrix.shape[1]):
                self._add_colored_edge(graph, mapped_matrix, (i, j))

        # Relabel nodes for drawing
        nodes_mapping = dict(list(enumerate(self.graphs["connection"]["nodes"])))
        graph = nx.relabel_nodes(graph, nodes_mapping)

        return graph

    def _add_colored_edge(
        self,
        graph: nx.Graph,
        mapped_adjacency_matrix: NDArray[np.float_],
        edge: Tuple[int, int],
    ) -> None:
        """Give an edge of the graph a color based on the mapping. Gray edges are
        unused, green edges are mapped correctly and red edges need at least on swap.

        :param graph: Graph of which the edges must be colored.
        :param mapped_adjacency_matrix: Adjacency matrix of the mapped graph.
        :param edge: Edge to color.
        """
        is_connected = self.graphs["connection"]["matrix"][edge] != 0
        is_mapped = mapped_adjacency_matrix[edge] != 0
        if not is_connected and is_mapped:
            graph.add_edge(*edge, color="red")
        if is_connected and is_mapped:
            graph.add_edge(*edge, color="green")
        if is_connected and not is_mapped:
            graph.add_edge(*edge, color="gray")

    def _draw_connection_graph(self, screen: pygame.surface.Surface) -> None:
        """Draw the connection graph on subscreen1.

        :param screen: Screen to draw the connection graph on.
        """
        for u, v in self.graphs["connection"]["edges"]:
            pos_u = self.graphs["connection"]["render_positions"][u]
            pos_v = self.graphs["connection"]["render_positions"][v]
            self._draw_wide_line(screen, self.colors["basic_edge"], pos_u, pos_v)

        for x, y in self.graphs["connection"]["render_positions"].values():
            self._draw_point(int(x), int(y), screen)

    def _draw_interaction_graph(
        self, screen: pygame.surface.Surface, step: int, interaction_graph: nx.Graph
    ) -> None:
        """Draw the interaction graph on subscreen2.

        :param screen: Screen to draw the interaction graph on.
        :param step: Current step number.
        :param interaction_graph: ``networkx.Graph`` representation of the interaction
            graph to draw.
        """
        # If we don't have node positions for the interaction graph for some reason,
        # compute them. If we are at step 0 we should have a new interaction graph, and
        # we should also compute new positions.
        if step == 0 or len(self.graphs["interaction"]["render_positions"]) == 0:
            self.graphs["interaction"]["render_positions"] = self._get_render_positions(
                interaction_graph, self.subscreens[1]
            )

        for u, v in interaction_graph.edges():
            pos_u = self.graphs["interaction"]["render_positions"][u]
            pos_v = self.graphs["interaction"]["render_positions"][v]
            self._draw_wide_line(screen, self.colors["basic_edge"], pos_u, pos_v)

        for x, y in self.graphs["interaction"]["render_positions"].values():
            self._draw_point(int(x), int(y), screen)

    def _draw_mapped_graph(
        self, screen: pygame.surface.Surface, mapped_graph: nx.Graph
    ) -> None:
        """Draw the mapped graph on subscreen3.

        :param screen: Screen to draw the graph on.
        :param mapped_graph: ``networkx.Graph`` representation of the mapped graph. Each
            edge should have a color attached to it.
        """
        for u, v in mapped_graph.edges():
            pos_u = self.graphs["mapped"]["render_positions"][u]
            pos_v = self.graphs["mapped"]["render_positions"][v]
            if mapped_graph.edges[u, v]["color"] == "red":
                color = self.colors["missing_edge"]
            if mapped_graph.edges[u, v]["color"] == "green":
                color = self.colors["used_edge"]
            if mapped_graph.edges[u, v]["color"] == "gray":
                color = self.colors["unused_edge"]
            self._draw_wide_line(screen, color, pos_u, pos_v)

        for x, y in self.graphs["mapped"]["render_positions"].values():
            self._draw_point(int(x), int(y), screen)

    def _draw_point(self, x: int, y: int, screen: pygame.surface.Surface) -> None:
        """Draw a point on the screen.

        :param x: x coordinate of the point.
        :param y: y coordinate of the point.
        :param screen: Screen to add the point to.
        """
        gfxdraw.aacircle(screen, x, y, 10, self.colors["nodes"])
        gfxdraw.filled_circle(screen, x, y, 10, self.colors["nodes"])

    def _draw_wide_line(
        self,
        screen: pygame.surface.Surface,
        color: Tuple[int, int, int],
        p1: NDArray[np.float_],
        p2: NDArray[np.float_],
        *,
        width: int = 2,
    ) -> None:
        """Draw a wide line on the screen.

        :param screen: Screen to draw the line on.
        :param color: Color of the line.
        :param p1: Coordinates of the starting point of the line.
        :param p2: Coordinates of the end point of the line.
        :param width: Width of the line. Defaults to 2.
        """
        # distance between the points
        dis = np.linalg.norm(p2 - p1)

        # scaled perpendicular vector (vector from p1 & p2 to the polygon's points)
        sp = np.array([p1[1] - p2[1], p2[0] - p1[0]]) * 0.5 * width / dis

        # points
        points = (p1 - sp, p1 + sp, p2 + sp, p2 - sp)

        # draw the polygon
        pygame.gfxdraw.aapolygon(screen, points, color)  # type: ignore[arg-type]
        pygame.gfxdraw.filled_polygon(screen, points, color)  # type: ignore[arg-type]

    def _draw_header(
        self, text: str, subscreen: pygame.Rect, screen: pygame.surface.Surface
    ) -> None:
        """Draw a header above a subscreen.

        :param text: Text of the header.
        :param subscreen: Subscreen to draw the header above.
        :param screen: Main screen to draw on.
        """
        pygame_text = self.font["header"].render(text, True, self.colors["text"])
        text_center = (subscreen.center[0], subscreen.y - self.header_spacing / 2)
        text_position = pygame_text.get_rect(center=text_center)
        screen.blit(pygame_text, text_position)

    @staticmethod
    def _get_render_positions(
        graph: nx.Graph, subscreen: pygame.Rect
    ) -> Dict[Any, NDArray[np.float_]]:
        """Give the positions of the nodes of a graph on a given subscreen.

        :param graph: Graph of which the node positions must be determined.
        :param subscreen: the subscreen on which the graph will be drawn.
        :return: a dictionary where the keys are the names of the nodes, and the
            values are the coordinates of these nodes.
        """
        node_positions: Dict[Any, NDArray[np.float_]]
        node_positions = nx.spring_layout(graph, threshold=1e-6)

        # Scale and move the node positions to be centered on the subscreen
        for node, position in node_positions.items():
            node_positions[node] = position * 0.45 * subscreen.size + subscreen.center

        return node_positions

    @property
    def header_spacing(self) -> float:
        """Header spacing."""
        return self.font_size / 3 * 4
