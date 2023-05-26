"""This module contains a class used for rendering a ``Routing`` environment."""
from copy import deepcopy
from typing import Any, Dict, Tuple, Union

import networkx as nx
import numpy as np
import pygame
from networkx import Graph
from numpy.typing import ArrayLike, NDArray
from pygame import gfxdraw

from qgym.envs.routing.routing_state import RoutingState
from qgym.templates.visualiser import Visualiser

# Define some colors used during rendering
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
RED = (189, 18, 33)
GREEN = (174, 168, 0)
BLUE = (113, 164, 195)

# Type Alias
Color = Tuple[int, int, int]


class RoutingVisualiser(Visualiser):
    """Visualiser class for the ``Routing`` environment."""

    def __init__(self, connection_graph: Graph) -> None:
        """Init of the ``RoutingVisualiser``.

        :param connection_graph: ``networkx.Graph`` representation of the connection
            graph.
        """
        # Rendering data
        self.screen_dimensions = (1600, 700)
        self.font_size = 30

        self.screens: Dict[str, pygame.surface.Surface] = {}
        self.font: Dict[str, pygame.font.Font] = {}

        self.colors = {
            "node": BLUE,
            "edge": BLACK,
            "good gate": BLUE,
            "bad gate": RED,
            "text": BLACK,
            "circuit lines": BLACK,
            "background": WHITE,
            "hidden": BLACK,
            "mapping": WHITE,
        }

        # Save everything we need to know about the connection graph
        self.graph = {
            "render_positions": self._get_render_positions(connection_graph),
            "nodes": connection_graph.nodes,
            "edges": connection_graph.edges,
        }

    def _start_subscreens(
        self, screen: pygame.surface.Surface, padding: int = 20
    ) -> Tuple[pygame.surface.Surface, pygame.surface.Surface]:
        """Initialize the subscreens.

        :param screen: The parent screen.
        :param padding: The padding to be used in between the subscreens.
        """
        position_circuit = (0, self.header_spacing)
        width_circuit = self.screen_width * 0.75 - 0.5 * padding
        height_circuit = self.screen_height - self.header_spacing
        rect_circuit = pygame.Rect(position_circuit, (width_circuit, height_circuit))
        subscreen_circuit = screen.subsurface(rect_circuit)

        position_graph = (width_circuit + padding, self.header_spacing)
        width_graph = self.screen_width * 0.25 - 0.5 * padding
        height_graph = self.screen_height - self.header_spacing
        rect_graph = pygame.Rect(position_graph, (width_graph, height_graph))
        subscreen_graph = screen.subsurface(rect_graph)
        return subscreen_circuit, subscreen_graph

    def render(self, state: RoutingState, mode: str) -> Union[bool, NDArray[np.int_]]:
        """Render the current state using ``pygame``.

        :param state: State to render.
        :param mode: Mode to start pygame for ("human" and "rgb_array" are supported).
        :raise ValueError: When an invalid mode is provided.
        :return: In 'human' mode returns a boolean value encoding whether the ``pygame``
            screen is open. In 'rgb_array' mode returns an RGB array encoding of the
            rendered image.
        """
        if not self.screens:
            screen = self._start_screen("Routing Environment", mode)
            circuit_screen, graph_screen = self._start_subscreens(screen)
            self.screens = dict(main=screen, circuit=circuit_screen, graph=graph_screen)
            pygame.font.init()

        if len(self.font) == 0:
            self.font["header"] = pygame.font.SysFont("Arial", self.font_size)
            self.font["circuit"] = pygame.font.SysFont("Arial", 24)

        pygame.time.delay(10)

        self.screens["main"].fill(self.colors["background"])

        self._draw_connection_graph()
        self._draw_interaction_circuit(state)

        self._draw_header("Interaction Circuit", "circuit")
        self._draw_header("Connection Graph", "graph")

        return self._display(mode)

    def _draw_interaction_circuit(self, state: RoutingState) -> None:
        """Draw the interaction circuit on the interaction circuit subscreen."""
        x_text = self.screens["circuit"].get_width() * 0.05
        x_left = self.screens["circuit"].get_width() * 0.1
        x_right = self.screens["circuit"].get_width() * 0.95
        y_distance = self.screens["circuit"].get_height() / (state.n_qubits)
        y_lines = [0.5 * y_distance + y_distance * n for n in range(state.n_qubits)]
        for n, y in enumerate(y_lines):
            self._draw_wide_line(
                self.screens["circuit"],
                self.colors["circuit lines"],
                (x_left, y),
                (x_right, y),
            )
            pygame_text = self.font["header"].render(f"Q{n}", True, self.colors["text"])
            text_position = pygame_text.get_rect(center=(x_text, y))
            self.screens["circuit"].blit(pygame_text, text_position)

        dx_gates = (x_right - x_left) / state.max_interaction_gates
        x_gates = [
            x_left + dx_gates * (0.5 + n) for n in range(state.max_interaction_gates)
        ]
        for n, (qubit1, qubit2) in enumerate(state.interaction_circuit):
            physical_qubit1 = state.current_mapping[qubit1]
            physical_qubit2 = state.current_mapping[qubit2]
            if (physical_qubit1, physical_qubit2) in self.graph["edges"]:
                color = self.colors["good gate"]
            else:
                color = self.colors["bad gate"]

            point1 = (x_gates[n], y_lines[qubit1])
            point2 = (x_gates[n], y_lines[qubit2])
            self._draw_wide_line(self.screens["circuit"], color, point1, point2)
            self._draw_point(self.screens["circuit"], *point1, color)
            self._draw_point(self.screens["circuit"], *point2, color)

        shade_left_width = (state.position + 0.5) * dx_gates
        shade_left_height = self.screens["circuit"].get_height()
        shade_left = pygame.Surface((shade_left_width, shade_left_height))
        shade_left.set_alpha(128)
        shade_left.fill(self.colors["hidden"])
        self.screens["circuit"].blit(shade_left, (x_left - 0.5 * dx_gates, 0))

        shade_right_width = (
            state.max_interaction_gates - state.max_observation_reach - state.position
        ) * dx_gates
        shade_right_height = self.screens["circuit"].get_height()
        shade_right = pygame.Surface((shade_right_width, shade_right_height))
        shade_right.set_alpha(128)
        shade_right.fill(self.colors["hidden"])
        self.screens["circuit"].blit(shade_right, (x_right - shade_right_width, 0))

        mapping = np.arange(state.n_qubits)
        idx = 0
        for i in range(state.position):
            # Update the mapping before drawing it if a swap was applied at this position
            while (
                idx < len(state.swap_gates_inserted)
                and i == state.swap_gates_inserted[idx][0]
            ):
                _, qubit1, qubit2
                mapping[qubit1], mapping[qubit2] = mapping[qubit2], mapping[qubit1]
                idx += 1

            # Draw the mapping
            x = x_gates[i] - 0.5 * dx_gates
            for n, y in zip(mapping, y_lines):
                pygame_text = self.font["circuit"].render(
                    str(n), True, self.colors["mapping"]
                )
                text_position = pygame_text.get_rect(center=(x, y))
                self.screens["circuit"].blit(pygame_text, text_position)

    def _draw_connection_graph(self) -> None:
        """Draw the connection graph on the graph subscreen."""
        for u, v in self.graph["edges"]:
            pos_u = self.graph["render_positions"][u]
            pos_v = self.graph["render_positions"][v]
            self._draw_wide_line(
                self.screens["graph"], self.colors["edge"], pos_u, pos_v
            )

        for x, y in self.graph["render_positions"].values():
            self._draw_point(self.screens["graph"], x, y, self.colors["node"], 20)

    def _draw_point(
        self,
        screen: pygame.surface.Surface,
        x: float,
        y: float,
        color: Color,
        r: int = 10,
    ) -> None:
        """Draw a point on the screen.

        :param x: x coordinate of the point. Non integer values will be rounded down to
            the nearest integer.
        :param y: y coordinate of the point. Non integer values will be rounded down to
            the nearest integer.
        :param screen: Screen to add the point to.
        :param r: Radius of the point (in pixels). Defaults to 10.
        """
        gfxdraw.aacircle(screen, int(x), int(y), r, color)
        gfxdraw.filled_circle(screen, int(x), int(y), r, color)

    def _draw_wide_line(
        self,
        screen: pygame.surface.Surface,
        color: Color,
        p1: ArrayLike,
        p2: ArrayLike,
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
        dis = np.linalg.norm(np.asarray(p2) - np.asarray(p1))

        # scaled perpendicular vector (vector from p1 & p2 to the polygon's points)
        sp = np.asarray([p1[1] - p2[1], p2[0] - p1[0]]) * 0.5 * width / dis

        # points
        points = (p1 - sp, p1 + sp, p2 + sp, p2 - sp)

        # draw the polygon
        pygame.gfxdraw.aapolygon(screen, points, color)  # type: ignore[arg-type]
        pygame.gfxdraw.filled_polygon(screen, points, color)  # type: ignore[arg-type]

    def _draw_header(self, text: str, screen_name: str) -> None:
        """Draw a header above a subscreen.

        :param text: Text of the header.
        :param screen_name: Name of the subscreen, choose from 'graph' and 'circuit'.
        """
        pygame_text = self.font["header"].render(text, True, self.colors["text"])
        offset = self.screens[screen_name].get_offset()
        rect = self.screens[screen_name].get_rect(topleft=offset)
        text_center = (rect.center[0], rect.y - self.header_spacing / 2)
        text_position = pygame_text.get_rect(center=text_center)
        self.screens["main"].blit(pygame_text, text_position)

    def _get_render_positions(
        self, graph: nx.Graph, padding: int = 20
    ) -> Dict[Any, NDArray[np.float_]]:
        """Give the positions of the nodes of a graph on a given screen.

        :param graph: Graph of which the node positions must be determined.
        :param screen: the subscreen on which the graph will be drawn.
        :return: a dictionary where the keys are the names of the nodes, and the
            values are the coordinates of these nodes.
        """
        node_positions: Dict[Any, NDArray[np.float_]]
        node_positions = nx.spring_layout(graph, threshold=1e-6)

        # Scale and move the node positions to be centered on the graph subscreen
        width_graph_screen = self.screen_width * 0.25 - 0.5 * padding
        height_graph_screen = self.screen_height - self.header_spacing
        size = np.array([width_graph_screen, height_graph_screen])
        for node, position in node_positions.items():
            node_positions[node] = position * 0.45 * size + 0.5 * size

        return node_positions

    @property
    def is_open(self) -> bool:
        """Boolean value stating whether a ``pygame.screen`` is currently open."""
        return not self.screens

    @property
    def header_spacing(self) -> float:
        """Header spacing."""
        return self.font_size / 3 * 4


if __name__ == "__main__":
    from time import sleep

    from qgym.spaces import MultiDiscrete

    action_space = MultiDiscrete([2, 4, 4])

    graph = nx.Graph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 0)

    state = RoutingState(
        max_interaction_gates=10,
        max_observation_reach=5,
        connection_graph=graph,
        observation_booleans_flag=False,
        observation_connection_flag=False,
    )

    vis = RoutingVisualiser(graph)
    for _ in range(100):
        vis.render(state, "human")
        action = action_space.sample()
        state.update_state(action)
        print(action)
        sleep(0.1)
