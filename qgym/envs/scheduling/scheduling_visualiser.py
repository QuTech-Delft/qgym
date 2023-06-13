"""This module contains a class used for rendering the ``Scheduling`` environment."""
from typing import Dict, Tuple, Union, cast

import numpy as np
import pygame
from numpy.typing import NDArray

from qgym.custom_types import Gate
from qgym.envs.scheduling.scheduling_state import SchedulingState
from qgym.templates.visualiser import Visualiser
from qgym.utils.visualisation.colors import BLACK, BLUE, DARK_BLUE, WHITE
from qgym.utils.visualisation.typing import Font, Surface
from qgym.utils.visualisation.wrappers import write_text


class SchedulingVisualiser(Visualiser):
    """Visualiser class for the ``Scheduling`` environment."""

    def __init__(self, initial_state: SchedulingState) -> None:
        """Init of the ``SchedulingVisualiser``.

        :param initial_state: ``SchedulingState`` object containing the initial state of
            the environment to visualise.
        """
        # Rendering data
        self.screen = None
        self.screen_dimensions = (1500, 800)
        self.offset = {"x-axis": 100, "y-axis": 0}
        self.colors = {
            "gate_fill": BLUE,
            "gate_outline": DARK_BLUE,
            "gate_text": WHITE,
            "background": WHITE,
            "qubits": BLACK,
            "y-axis": BLACK,
        }

        n_qubits = initial_state.machine_properties.n_qubits
        gate_height = (self.screen_height - self.offset["y-axis"]) / n_qubits

        longest_gate = 0
        for n_cycles in initial_state.machine_properties.gates.values():
            longest_gate = max(longest_gate, n_cycles)

        self.gate_size_info = {"height": gate_height, "longest": longest_gate}

        # define attributes that are set later
        self.font: Dict[str, Font] = {}
        self._cycle_width = 0.0

    def render(
        self, state: SchedulingState, mode: str
    ) -> Union[bool, NDArray[np.int_]]:
        """Render the current state using pygame.

        :param mode: The mode to render with (supported modes are found in
            `self.metadata`.).
        :raise ValueError: If an unsupported mode is provided.
        :return: Result of rendering.
        """
        if self.screen is None:
            self.screen = self._start_screen("Scheduling Environment", mode)

        if len(self.font) == 0:
            gate_font, axis_font = self._start_font()
            self.font["gate"] = gate_font
            self.font["axis"] = axis_font

        pygame.time.delay(50)

        self.screen.fill(self.colors["background"])
        self._draw_y_axis(state.machine_properties.n_qubits)

        self._cycle_width = (self.screen_width - self.offset["x-axis"]) / (
            state.cycle + self.gate_size_info["longest"]
        )

        for gate_idx, scheduled_cycle in enumerate(state.circuit_info.schedule):
            if scheduled_cycle != -1:
                gate = state.circuit_info.encoded[gate_idx]
                gate_cycle_length = cast(
                    Dict[int, int], state.machine_properties.gates
                )[gate.name]
                gate_name = state.utils.gate_encoder.decode_gates(gate.name)
                self._draw_scheduled_gate(
                    gate, scheduled_cycle, gate_cycle_length, gate_name
                )

        return self._display(mode)

    def _draw_y_axis(self, n_qubits: int) -> None:
        """Draw the y-axis of the display.

        :param n_qubits: Number of qubits of the machine.
        """
        screen = cast(Surface, self.screen)
        for i in range(n_qubits):
            pos = (
                self.offset["x-axis"] / 2,
                self.gate_size_info["height"] * (n_qubits - i - 0.5),
            )
            write_text(screen, self.font["axis"], f"Q{i}", pos, self.colors["y-axis"])

    def _draw_scheduled_gate(
        self, gate: Gate, scheduled_cycle: int, gate_cycle_length: int, gate_name: str
    ) -> None:
        """Draw a gate on the screen.

        :param gate: Gate to draw.
        :param scheduled_cycle: Cycle the gate is scheduled.
        :param gate_cycle_length: Length of the gat in machine cycles.
        :param gate_name: Name of the gate.
        """
        self._draw_gate_block(gate_name, gate_cycle_length, gate.q1, scheduled_cycle)
        if gate.q1 != gate.q2:
            self._draw_gate_block(
                gate_name, gate_cycle_length, gate.q2, scheduled_cycle
            )

    def _draw_gate_block(
        self, gate_name: str, gate_cycle_length: int, qubit: int, scheduled_cycle: int
    ) -> None:
        """Draw a single block of a gate (gates can consist of 1 or 2 blocks).

        :param gate_name: Name of the gate.
        :param gate_cycle_length: Length of the gat in machine cycles.
        :param qubit: Qubit in which the gate acts.
        :param scheduled_cycle: Cycle in which the gate is scheduled.
        """
        screen = cast(Surface, self.screen)
        gate_width = self._cycle_width * gate_cycle_length
        gate_box_size = (0.98 * gate_width, 0.98 * self.gate_size_info["height"])

        box_pos = (
            self.screen_width - scheduled_cycle * self._cycle_width - gate_width,
            self.screen_height
            - qubit * self.gate_size_info["height"]
            - self.offset["y-axis"]
            - self.gate_size_info["height"],
        )
        gate_box = pygame.Rect(box_pos, gate_box_size)

        pygame.draw.rect(screen, self.colors["gate_fill"], gate_box, border_radius=5)
        pygame.draw.rect(
            screen, self.colors["gate_outline"], gate_box, width=2, border_radius=5
        )
        write_text(
            screen=screen,
            font=self.font["gate"],
            text=gate_name.upper(),
            pos=gate_box.center,
            color=self.colors["gate_text"],
        )

    def _start_font(self) -> Tuple[Font, Font]:
        """Start the pygame fonts for the gate and axis font.

        :return: pygame fonts for the gate and axis font.
        """
        pygame.font.init()
        gate_font = pygame.font.SysFont("Arial", 12)
        axis_font = pygame.font.SysFont("Arial", 30)
        return gate_font, axis_font
