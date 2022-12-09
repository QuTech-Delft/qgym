"""This module contains a class used for rendering the ``Scheduling`` environment."""
from typing import Any, Mapping, Optional, Tuple

import pygame
from pygame.font import Font

from qgym._visualiser import Visualiser
from qgym.custom_types import Gate
from qgym.utils import GateEncoder

# Define some colors used during rendering
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (71, 115, 147)
BLUE = (113, 164, 195)


class SchedulingVisualiser(Visualiser):
    """Visualiser class for the ``Scheduling`` environment."""

    def __init__(
        self,
        *,
        gate_encoder: GateEncoder,
        gate_cycle_length: Mapping[int, int],
        n_qubits: int,
    ) -> None:
        """Init of the ``SchedulingVisualiser``.

        :param gate_encoder: ``GateEncoder`` object of a ``Scheduling`` environment.
        :param gate_cycle_length: ``Mapping`` of cycle lengths for the gates of the
            scheduling environment.
        :param n_qubits: Number of qubits of the scheduling environment.
        """
        # Rendering data
        self.screen = None
        self.is_open = False
        self.screen_width = 1500
        self.screen_height = 800
        self.x_axis_offset = 100
        self.y_axis_offset = 0

        self.colors = {
            "gate_fill": BLUE,
            "gate_outline": DARK_BLUE,
            "gate_text": WHITE,
            "background": WHITE,
            "qubits": BLACK,
        }

        subscreen_pos = (self.x_axis_offset, 0)
        subscreen_size = (
            self.screen_width - self.x_axis_offset,
            self.screen_height - self.y_axis_offset,
        )
        self.subscreen = pygame.Rect(subscreen_pos, subscreen_size)

        self._gate_encoder = gate_encoder
        self._gate_cycle_length = gate_cycle_length
        self._n_qubits = n_qubits
        self._gate_height = self.subscreen.height / self._n_qubits

        self._longest_gate = 0
        for n_cycles in gate_cycle_length.values():
            self._longest_gate = max(self._longest_gate, n_cycles)

        # define attributes that are set later
        self.font: Optional[Font] = None
        self.axis_font: Optional[Font] = None
        self._cycle_width = 0

    def render(self, state: Mapping[str, Any], mode: str) -> Any:
        """Render the current state using pygame.

        :param mode: The mode to render with (supported modes are found in
            `self.metadata`.).
        :raise ValueError: If an unsupported mode is provided.
        :return: Result of rendering.
        """
        if self.screen is None:
            self.screen = self._start_screen("Scheduling Environment", mode)

        if self.font is None or self.axis_font is None:
            self.font, self.axis_font = self._start_font()

        encoded_circuit = state["encoded_circuit"]

        pygame.time.delay(50)

        self.screen.fill(self.colors["background"])
        self._draw_y_axis(self.colors["qubits"], self.screen, self.axis_font)

        self._cycle_width = self.subscreen.width / (state["cycle"] + self._longest_gate)

        for gate_idx, scheduled_cycle in enumerate(state["schedule"]):
            if scheduled_cycle != -1:
                self._draw_scheduled_gate(
                    encoded_circuit[gate_idx], scheduled_cycle, self.screen, self.font
                )

        return self._display(mode)

    def _draw_y_axis(
        self,
        color: Tuple[int, int, int],
        screen: pygame.surface.Surface,
        axis_font: Font,
    ) -> None:
        """Draw the y-axis of the display.

        :param color: Color of the y-axis.
        :param screen: Screen to draw the y-axis on.
        :param axis_font: Font object to use for the axis labels.
        """
        for i in range(self._n_qubits):
            text = axis_font.render(f"Q{i}", True, color)
            text_center = (
                self.x_axis_offset / 2,
                self._gate_height * (self._n_qubits - i - 0.5),
            )
            text_position = text.get_rect(center=text_center)
            screen.blit(text, text_position)

    def _draw_scheduled_gate(
        self,
        gate: Gate,
        scheduled_cycle: int,
        screen: pygame.surface.Surface,
        font: Font,
    ) -> None:
        """Draw a gate on the screen.

        :param gate: Gate to draw.
        :param scheduled_cycle: Cycle the gate is scheduled.
        :param screen: Screen to draw the gate on.
        :param font: Font object to write the text with.
        """
        self._draw_gate_block(gate.name, gate.q1, scheduled_cycle, screen, font)
        if gate.q1 != gate.q2:
            self._draw_gate_block(gate.name, gate.q2, scheduled_cycle, screen, font)

    def _draw_gate_block(
        self,
        gate_int_name: int,
        qubit: int,
        scheduled_cycle: int,
        screen: pygame.surface.Surface,
        font: Font,
    ) -> None:
        """Draw a single block of a gate (gates can consist of 1 or 2 blocks).

        :param gate_int_name: Integer encoding of the gate name.
        :param qubit: Qubit in which the gate acts.
        :param scheduled_cycle: Cycle in which the gate is scheduled.
        :param screen: Screen to draw the gate block on.
        :param font: Font object to write the text with.
        """
        gate_width = self._cycle_width * self._gate_cycle_length[gate_int_name]
        gate_box_size = (0.98 * gate_width, 0.98 * self._gate_height)

        box_pos = (
            self.screen_width - scheduled_cycle * self._cycle_width - gate_width,
            self.screen_height
            - qubit * self._gate_height
            - self.y_axis_offset
            - self._gate_height,
        )
        gate_box = pygame.Rect(box_pos, gate_box_size)

        pygame.draw.rect(screen, self.colors["gate_fill"], gate_box, border_radius=5)
        pygame.draw.rect(
            screen, self.colors["gate_outline"], gate_box, width=2, border_radius=5
        )

        gate_name: str
        gate_name = self._gate_encoder.decode_gates(gate_int_name)
        text = font.render(gate_name.upper(), True, self.colors["gate_text"])
        text_position = text.get_rect(center=gate_box.center)
        screen.blit(text, text_position)

    def _start_font(self) -> Tuple[Font, Font]:
        """Start the pygame fonts for the header and axis font.

        :return: pygame fonts for the header and axis font.
        """
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 12)
        axis_font = pygame.font.SysFont("Arial", 30)

        self.is_open = True
        return font, axis_font
