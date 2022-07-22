"""This module contains a class used for rendering the scheduling environment."""

from numbers import Integral
from typing import Any, Mapping

import pygame

from qgym.utils import GateEncoder

# Define some colors used during rendering
BACKGROUND_COLOR = (150, 150, 150)  # Gray
TEXT_COLOR = (225, 225, 225)  # White
GATE_COLOR = (0, 0, 0)  # Black


class SchedulingVisualiser:
    """Visualiser class for the scheduling environment"""

    def __init__(
        self,
        *,
        gate_encoder: GateEncoder,
        gate_cycle_length: Mapping[Integral, Integral],
        n_qubits: Integral
    ) -> None:
        """Initialize the visualiser.

        :param GateEncoder: GateEncoder object of the scheduling environment.
        :param gate_cycle_length: Mapping of cycle lengths for the gates of the
            scheduling environment.
        :param n_qubits: number of qubits of the scheduling environment."""

        # Rendering data
        self.screen = None
        self.is_open = False
        self.screen_width = 1500
        self.screen_height = 800

        self._gate_encoder = gate_encoder
        self._gate_cycle_length = gate_cycle_length
        self._n_qubits = n_qubits
        self._gate_height = self.screen_height / self._n_qubits

    def render(self, state: Mapping[str, Any]) -> bool:
        """Render the current state using pygame.

        :param mode: The mode to render with (default is 'human')."""

        if self.screen is None:
            self.start()

        self._encoded_circuit = state["encoded_circuit"]

        self.screen.fill(BACKGROUND_COLOR)

        pygame.time.delay(10)

        self._cycle_width = self.screen_width / (state["cycle"] + 10)

        for gate_idx, scheduled_cycle in enumerate(state["schedule"]):
            if scheduled_cycle != -1:
                self._draw_scheduled_gate(gate_idx, scheduled_cycle)

        pygame.event.pump()
        pygame.display.flip()

        return self.is_open

    def _draw_scheduled_gate(
        self, gate_idx: Integral, scheduled_cycle: Integral
    ) -> None:
        """Draw a gate on the screen.

        :param gate_idx: index of the gate to draw.
        :param scheduled_cycle: cycle the gate is scheduled."""

        gate = self._encoded_circuit[gate_idx]

        self._draw_gate_block(gate.name, gate.q1, scheduled_cycle)
        if gate.q1 != gate.q2:
            self._draw_gate_block(gate.name, gate.q2, scheduled_cycle)

    def _draw_gate_block(
        self, gate_intname: Integral, qubit: Integral, scheduled_cycle: Integral
    ) -> None:
        """Draw a single block of a gate (gates can consist of 1 or 2 blocks).

        :param gate_intname: integer encoding of the gate name.
        :param qubit: qubit in which the gate acts.
        :param scheduled_cycle: cycle in which the gate is scheduled."""

        gate_width = self._cycle_width * self._gate_cycle_length[gate_intname]

        gate_box = pygame.Rect(0, 0, gate_width, self._gate_height)
        box_x = self.screen_width - scheduled_cycle * self._cycle_width
        box_y = self.screen_height - qubit * self._gate_height
        gate_box.bottomright = (box_x, box_y)

        pygame.draw.rect(self.screen, GATE_COLOR, gate_box)

        gate_name = self._gate_encoder.decode_gates(gate_intname)
        text = self.font.render(gate_name.upper(), True, TEXT_COLOR)
        text_postition = text.get_rect(center=gate_box.center)
        self.screen.blit(text, text_postition)

    def start(self) -> None:
        """Start pygame."""

        pygame.display.init()
        pygame.display.set_caption("Scheduling Environment")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)

        self.is_open = True

    def close(self) -> None:
        """Close the screen used for rendering."""

        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()
            self.is_open = False
            self.screen = None
