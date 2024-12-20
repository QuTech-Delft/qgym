"""This module contains the :class:`InitialMappingVisualiser` class.

:class:`InitialMappingVisualiser` is used for rendering a
:class:`~qgym.envs.InitialMapping` environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pygame

from qgym.templates.visualiser import RenderData, Visualiser
from qgym.utils.visualisation.colors import BLACK, BLUE, GRAY, GREEN, RED, WHITE
from qgym.utils.visualisation.wrappers import draw_point, draw_wide_line

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qgym.envs.initial_mapping.initial_mapping_state import InitialMappingState
    from qgym.utils.visualisation.typing import Font, Surface

# pylint: disable=invalid-name


class InitialMappingVisualiser(Visualiser):
    """Visualiser class for the :class:`~qgym.envs.InitialMapping` environment."""

    def __init__(self, render_mode: str, connection_graph: nx.Graph) -> None:
        # pylint: disable=line-too-long
        """Init of the :class:`~qgym.envs.initial_mapping.InitialMappingVisualiser`.

        Args:
            connection_graph: `networkx Graph <https://networkx.org/documentation/stable/reference/classes/graph.html>`_
                representation of the connection graph.
            render_mode: If 'human' open a ``pygame`` screen visualizing the step. If
                'rgb_array', return an RGB array encoding of the rendered frame on each
                render call.
        """
        # pylint: enable=line-too-long
        # Rendering data
        colors = {
            "nodes": BLUE,
            "basic_edge": BLACK,
            "unused_edge": GRAY,
            "used_edge": GREEN,
            "missing_edge": RED,
            "text": BLACK,
            "background": WHITE,
        }
        self.render_data = RenderData(
            screen=self._start_screen("Mapping Environment", render_mode, (1300, 730)),
            font=self._start_font(),
            colors=colors,
            render_mode=render_mode,
        )

        self.subscreens = self._init_subscreen_rectangles()

        # Save everything we need to know about the graphs
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
    ) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        """Initialize the ``pygame.Rect`` objects used for drawing the subscreens.

        Args:
            padding: The padding to be used inbetween the subscreens.

        Returns:
            A tuple containing two subscreens.
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

    def render(self, state: InitialMappingState) -> NDArray[np.int_] | None:
        """Render the current state using ``pygame``.

        The upper left screen shows the connection graph. The lower left screen the
        interaction graph. The right screen shows the mapped graph. Gray edges are
        unused, green edges are mapped correctly and red edges need at least on swap.

        Args:
            state: State to render.

        Raises:
            ValueError: When an invalid mode is provided.

        Returns:
            In 'human' mode returns a boolean value encoding whether the ``pygame``
            screen is open. In 'rgb_array' mode returns an RGB array encoding of the
            rendered image.
        """
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

        return self._display()

    def _get_mapped_graph(
        self, mapping: dict[int, int], interaction_graph_matrix: NDArray[np.float64]
    ) -> nx.Graph:
        """Construct a mapped graph.

        In this graph gray edges are unused, green edges are mapped correctly and red
        edges need at least on swap. This function is used during rendering.

        Args:
            mapping: Mapping dictionary of the state to render.
            interaction_graph_matrix: Interaction graph matrix of the
                current interaction graph.

        Returns:
            Mapped graph.
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
        return nx.relabel_nodes(graph, nodes_mapping)

    def _add_colored_edge(
        self,
        graph: nx.Graph,
        mapped_adjacency_matrix: NDArray[np.float64],
        edge: tuple[int, int],
    ) -> None:
        """Give an edge of the graph a color based on the mapping.

        Gray edges are unused, green edges are mapped correctly and red edges need at
        least on swap.

        Args:
            graph: Graph of which the edges must be colored.
            mapped_adjacency_matrix: Adjacency matrix of the mapped graph.
            edge: Edge to color.
        """
        is_connected = self.graphs["connection"]["matrix"][edge] != 0
        is_mapped = mapped_adjacency_matrix[edge] != 0
        if not is_connected and is_mapped:
            graph.add_edge(*edge, color="red")
        if is_connected and is_mapped:
            graph.add_edge(*edge, color="green")
        if is_connected and not is_mapped:
            graph.add_edge(*edge, color="gray")

    def _draw_connection_graph(self, screen: Surface) -> None:
        """Draw the connection graph on subscreen1.

        Args:
            screen: Screen to draw the connection graph on.
        """
        for u, v in self.graphs["connection"]["edges"]:
            pos_u = self.graphs["connection"]["render_positions"][u]
            pos_v = self.graphs["connection"]["render_positions"][v]
            draw_wide_line(screen, self.colors["basic_edge"], pos_u, pos_v)

        for pos in self.graphs["connection"]["render_positions"].values():
            draw_point(screen, pos, self.colors["nodes"])

    def _draw_interaction_graph(
        self, screen: Surface, step: int, interaction_graph: nx.Graph
    ) -> None:
        """Draw the interaction graph on subscreen2.

        Args:
            screen: Screen to draw the interaction graph on.
            step: Current step number.
            interaction_graph: ``networkx.Graph`` representation of the interaction
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
            draw_wide_line(screen, self.colors["basic_edge"], pos_u, pos_v)

        for pos in self.graphs["interaction"]["render_positions"].values():
            draw_point(screen, pos, self.colors["nodes"])

    def _draw_mapped_graph(self, screen: Surface, mapped_graph: nx.Graph) -> None:
        """Draw the mapped graph on subscreen3.

        Args:
            screen: Screen to draw the graph on.
            mapped_graph: ``networkx.Graph`` representation of the mapped graph. Each
                edge should have a color attached to it.
        """
        for u, v in mapped_graph.edges():
            pos_u = self.graphs["mapped"]["render_positions"][u]
            pos_v = self.graphs["mapped"]["render_positions"][v]
            if mapped_graph.edges[u, v]["color"] == "red":
                color = self.colors["missing_edge"]
            elif mapped_graph.edges[u, v]["color"] == "green":
                color = self.colors["used_edge"]
            elif mapped_graph.edges[u, v]["color"] == "gray":
                color = self.colors["unused_edge"]
            else:
                msg = "unknown color"
                raise ValueError(msg)
            draw_wide_line(screen, color, pos_u, pos_v)

        for pos in self.graphs["mapped"]["render_positions"].values():
            draw_point(screen, pos, self.colors["nodes"])

    def _draw_header(self, text: str, subscreen: pygame.Rect, screen: Surface) -> None:
        """Draw a header above a subscreen.

        Args:
            text: Text of the header.
            subscreen: Subscreen to draw the header above.
            screen: Main screen to draw on.
        """
        pygame_text = self.font["header"].render(
            text, antialias=True, color=self.colors["text"]
        )
        text_center = (subscreen.center[0], subscreen.y - self.header_spacing / 2)
        text_position = pygame_text.get_rect(center=text_center)
        screen.blit(pygame_text, text_position)

    @staticmethod
    def _get_render_positions(
        graph: nx.Graph, subscreen: pygame.Rect
    ) -> dict[Any, NDArray[np.float64]]:
        """Give the positions of the nodes of a graph on a given subscreen.

        Args:
            graph: Graph of which the node positions must be determined.
            subscreen: the subscreen on which the graph will be drawn.

        Returns:
            A dictionary where the keys are the names of the nodes, and the values are
            the coordinates of these nodes.
        """
        node_positions: dict[Any, NDArray[np.float64]]
        node_positions = nx.spring_layout(graph, threshold=1e-6)

        # Scale and move the node positions to be centered on the subscreen
        for node, position in node_positions.items():
            new_position = 0.45 * position * subscreen.size + subscreen.center
            node_positions[node] = np.asarray(new_position, dtype=np.float64)

        return node_positions

    def _start_font(self) -> dict[str, Font]:
        """Start the pygame font.

        Returns:
            pygame fonts for the gate and axis font.
        """
        pygame.font.init()
        return {"header": pygame.font.SysFont("Arial", 30)}

    @property
    def header_spacing(self) -> float:
        """Header spacing."""
        return 30 / 3 * 4
