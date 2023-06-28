"""This module contains a class used for rendering a ``Routing`` environment."""
from typing import Any, Deque, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pygame
from networkx import Graph
from numpy.typing import NDArray

from qgym.envs.routing.routing_state import RoutingState
from qgym.templates.visualiser import Visualiser
from qgym.utils.visualisation.colors import BLACK, BLUE, GRAY, RED, WHITE
from qgym.utils.visualisation.typing import Font, Surface
from qgym.utils.visualisation.wrappers import (
    draw_point,
    draw_wide_line,
    shade_rect,
    write_text,
)


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

        self.screens: Dict[str, Surface] = {}
        self.font: Dict[str, Font] = {}

        self.colors = {
            "node": BLUE,
            "node_labels": WHITE,
            "edge": BLACK,
            "good gate": BLUE,
            "bad gate": RED,
            "passed gate": GRAY,
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
        self, screen: Surface, padding: int = 20
    ) -> Tuple[Surface, Surface]:
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
            self.screens = {
                "main": screen,
                "circuit": circuit_screen,
                "graph": graph_screen,
            }
            pygame.font.init()

        if len(self.font) == 0:
            self._setup_fonts()

        pygame.time.delay(10)

        self.screens["main"].fill(self.colors["background"])

        self._draw_connection_graph()
        self._draw_interaction_circuit(state)

        self._draw_header("Interaction Circuit", "circuit")
        self._draw_header("Connection Graph", "graph")

        return self._display(mode)

    def _draw_interaction_circuit(self, state: RoutingState) -> None:
        """Draw the interaction circuit on the interaction circuit subscreen.

        :param state: Current state.
        """
        x_text = self.screens["circuit"].get_width() * 0.05
        x_left = self.screens["circuit"].get_width() * 0.1
        x_right = self.screens["circuit"].get_width() * 0.95
        y_distance = self.screens["circuit"].get_height() / (state.n_qubits)
        y_lines = y_distance * (0.5 + np.arange(state.n_qubits))
        dx_gates = (x_right - x_left) / state.max_interaction_gates
        x_gates = x_left + dx_gates * (0.5 + np.arange(state.max_interaction_gates))

        self._draw_circuit_lines(
            x_text=x_text, x_left=x_left, x_right=x_right, y_lines=y_lines
        )
        self._draw_interaction_gates(state=state, x_gates=x_gates, y_lines=y_lines)
        self._draw_observation_reach(state=state, x_left=x_left, x_right=x_right)
        self._draw_mapping(state=state, x_gates=x_gates, y_lines=y_lines)

    def _draw_circuit_lines(
        self,
        *,
        x_text: float,
        x_left: float,
        x_right: float,
        y_lines: NDArray[np.float_],
    ) -> None:
        """Draw the circuit lines on the 'circuit' screen and label them.

        :param x_text: x coordinate of the labels of the circuit lines.
        :param x_left: Left most x coordinate of the circuit lines.
        :param x_right: Right most x coordinate of the circuit lines.
        :param y_lines: Array of y coordinates of the circuit lines.
        """
        for qubit_idx, y_line in enumerate(y_lines):
            draw_wide_line(
                self.screens["circuit"],
                self.colors["circuit lines"],
                (x_left, y_line),
                (x_right, y_line),
            )
            write_text(
                screen=self.screens["circuit"],
                font=self.font["header"],
                text=f"Q{qubit_idx}",
                pos=(x_text, y_line),
                color=self.colors["text"],
            )

    def _draw_interaction_gates(
        self,
        *,
        state: RoutingState,
        x_gates: NDArray[np.float_],
        y_lines: NDArray[np.float_],
    ) -> None:
        """Draw the interaction gates on the 'circuit' screen.

        :param state: ``RoutingState`` to draw the interaction gates.
        :param x_gates: Array of x coordinates of the swap gates.
        :param y_lines: Array of y coordinates of the circuit lines.
        """
        for i, (qubit1, qubit2) in enumerate(state.interaction_circuit):
            physical_qubit1, physical_qubit2 = state.mapping[[qubit1, qubit2]]
            if i < state.position:
                color = self.colors["passed gate"]
            elif (physical_qubit1, physical_qubit2) in self.graph["edges"]:
                color = self.colors["good gate"]
            else:
                color = self.colors["bad gate"]

            point1 = (x_gates[i], y_lines[qubit1])
            point2 = (x_gates[i], y_lines[qubit2])
            draw_wide_line(self.screens["circuit"], color, point1, point2)
            draw_point(self.screens["circuit"], point1, color)
            draw_point(self.screens["circuit"], point2, color)

    def _draw_observation_reach(
        self, *, state: RoutingState, x_left: float, x_right: float
    ) -> None:
        """Draw shades on the 'circuit' screen to show the observation size.

        :param state: Current state to draw the observation reach of.
        :param x_left: Left most x coordinate of the circuit lines.
        :param x_right: Right most x coordinate of the circuit lines.
        """
        dx_gates = (x_right - x_left) / state.max_interaction_gates

        shade_left_width = (state.position + 0.5) * dx_gates
        shade_height = self.screens["circuit"].get_height()
        shade_rect(
            screen=self.screens["circuit"],
            size=(shade_left_width, shade_height),
            pos=(x_left - 0.25 * dx_gates, 0),
            color=self.colors["hidden"],
            alpha=128,
        )

        shade_right_width = (
            state.max_interaction_gates - state.max_observation_reach - state.position
        ) * dx_gates
        if shade_right_width > 0:
            shade_rect(
                screen=self.screens["circuit"],
                size=(shade_right_width, shade_height),
                pos=(x_right - shade_right_width, 0),
                color=self.colors["hidden"],
                alpha=128,
            )

    def _draw_mapping(
        self,
        *,
        state: RoutingState,
        x_gates: NDArray[np.float_],
        y_lines: NDArray[np.float_],
    ) -> None:
        """Draw the mapping on the 'circuit' screen.

        :param state: ``RoutingState`` to draw the mapping of.
        :param x_gates: Array of x coordinates of the swap gates.
        :param y_lines: Array of y coordinates of the circuit lines.
        """
        dx_gates = x_gates[1] - x_gates[0]
        dy_lines = y_lines[1] - y_lines[0]
        mapping = np.arange(state.n_qubits, dtype=int)
        starting_idx = 0
        for i in range(state.position):
            old_mapping = mapping.copy()
            starting_idx, mapping, n_swaps = self._update_mapping(
                mapping=mapping,
                swap_gates_inserted=state.swap_gates_inserted,
                position=i,
                starting_idx=starting_idx,
            )

            # Draw the mapping
            x_mapping = x_gates[i] - 0.5 * dx_gates
            for physical_qubit, y_logical_qubit, is_changed in zip(
                mapping, y_lines, old_mapping != mapping
            ):
                write_text(
                    screen=self.screens["circuit"],
                    font=self.font["mapping_emph" if is_changed else "mapping"],
                    text=str(physical_qubit),
                    pos=(x_mapping, y_logical_qubit),
                    color=self.colors["mapping"],
                )

            write_text(
                screen=self.screens["circuit"],
                font=self.font["n_swaps"],
                text=str(n_swaps),
                pos=(x_mapping, y_lines[0] - 0.3 * dy_lines),
                color=self.colors["mapping"],
            )

    def _update_mapping(
        self,
        *,
        mapping: NDArray[np.int_],
        swap_gates_inserted: Deque[Tuple[int, int, int]],
        position: int,
        starting_idx: int = 0,
    ) -> Tuple[int, NDArray[np.int_], int]:
        """Update the mapping to conform to the current position.

        :param mapping: Mapping of the previous position
        :param swap_gates_inserted: List of swap gates inserted.
        :position: Position in the interaction circuit where the mapping must be made.
        :starting_idx: Index of the last swap gate of the previous position.
        :returns: Tuple with a starting index for the next position, the updated
            mapping and number of swap gates used since the last mapping.
        """
        n_swaps = 0
        if starting_idx >= len(swap_gates_inserted):
            return starting_idx, mapping, n_swaps

        for i in range(starting_idx, len(swap_gates_inserted)):
            swap_position, qubit1, qubit2 = swap_gates_inserted[i]
            if position != swap_position:
                break

            n_swaps += 1
            mapping[[qubit1, qubit2]] = mapping[[qubit2, qubit1]]
        return i, mapping, n_swaps

    def _draw_connection_graph(self) -> None:
        """Draw the connection graph on the graph subscreen."""
        for node_u, node_v in self.graph["edges"]:
            pos_u = self.graph["render_positions"][node_u]
            pos_v = self.graph["render_positions"][node_v]
            draw_wide_line(self.screens["graph"], self.colors["edge"], pos_u, pos_v)

        for label, pos in self.graph["render_positions"].items():
            draw_point(self.screens["graph"], pos, self.colors["node"], 20)
            write_text(
                screen=self.screens["graph"],
                font=self.font["graph"],
                text=str(label),
                pos=pos,
                color=self.colors["node_labels"],
            )

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

    def _setup_fonts(self) -> None:
        """Setup the fonts for rendering with pygame."""
        self.font["header"] = pygame.font.SysFont("Arial", self.font_size)
        self.font["circuit"] = pygame.font.SysFont("Arial", 24)
        self.font["mapping"] = pygame.font.SysFont("Arial", 22)
        self.font["mapping_emph"] = pygame.font.SysFont(
            "Arial", 24, bold=True, italic=True
        )
        self.font["n_swaps"] = pygame.font.SysFont("Arial", 28)
        self.font["graph"] = pygame.font.SysFont("Arial", 24)

    @property
    def is_open(self) -> bool:
        """Boolean value stating whether a ``pygame.screen`` is currently open."""
        return not self.screens

    @property
    def header_spacing(self) -> float:
        """Header spacing."""
        return self.font_size / 3 * 4
