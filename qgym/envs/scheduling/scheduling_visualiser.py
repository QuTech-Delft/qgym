"""This module contains a class used for rendering the :class:`~qgym.envs.Scheduling`
environment.
"""

from __future__ import annotations

from typing import Dict, cast

import numpy as np
import pygame
from numpy.typing import NDArray

from qgym.custom_types import Gate
from qgym.envs.scheduling.scheduling_state import SchedulingState
from qgym.templates.visualiser import RenderData, Visualiser
from qgym.utils.visualisation.colors import BLACK, BLUE, DARK_BLUE, WHITE
from qgym.utils.visualisation.typing import Font
from qgym.utils.visualisation.wrappers import write_text


class SchedulingVisualiser(Visualiser):
    """Visualiser class for the :class:`~qgym.envs.Scheduling` environment."""

    def __init__(self, render_mode: str, initial_state: SchedulingState) -> None:
        """Init of the :class:`SchedulingVisualiser`.

        Args:
            initial_state: :class:`~qgym.envs.scheduling.SchedulingState` object
                containing the initial state of the environment to visualise.
            render_mode: If ``"human"`` open a ``pygame`` screen visualizing the step.
                If ``"rgb_array"``, return an RGB array encoding of the rendered frame
                on each render call.
        """
        # Rendering data
        self.offset = {"x-axis": 100, "y-axis": 0}
        colors = {
            "gate_fill": BLUE,
            "gate_outline": DARK_BLUE,
            "gate_text": WHITE,
            "background": WHITE,
            "qubits": BLACK,
            "y-axis": BLACK,
        }
        self.render_data = RenderData(
            screen=self._start_screen(
                "Scheduling Environment", render_mode, (1500, 800)
            ),
            font=self._start_font(),
            colors=colors,
            render_mode=render_mode,
        )

        n_qubits = initial_state.machine_properties.n_qubits
        gate_height = (self.screen_height - self.offset["y-axis"]) / n_qubits

        longest_gate = 0
        for n_cycles in initial_state.machine_properties.gates.values():
            longest_gate = max(longest_gate, n_cycles)

        self.gate_size_info = {"height": gate_height, "longest": longest_gate}

        # define attributes that are set later
        self._cycle_width = 0.0

    def render(self, state: SchedulingState) -> None | NDArray[np.int_]:
        """Render the current state using pygame.

        Args:
            state: State to render.

        Raises:
            ValueError: If an unsupported mode is provided.

        Returns:
            Result of rendering, based on `render_mode`.
        """
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
                    "Dict[int, int]", state.machine_properties.gates
                )[gate.name]
                gate_name = state.utils.gate_encoder.decode_gates(gate.name)
                self._draw_scheduled_gate(
                    gate, scheduled_cycle, gate_cycle_length, gate_name
                )

        return self._display()

    def _draw_y_axis(self, n_qubits: int) -> None:
        """Draw the y-axis of the display.

        Args:
            n_qubits: Number of qubits of the machine.
        """
        screen = self.screen
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

        Args:
            gate: Gate to draw.
            scheduled_cycle: Cycle the gate is scheduled.
            gate_cycle_length: Length of the gate in terms of machine cycles.
            gate_name: Name of the gate.
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

        Args:
            gate_name: Name of the gate.
            gate_cycle_length: Length of the gate in terms of machine cycles.
            qubit: Qubit in which the gate acts.
            scheduled_cycle: Cycle in which the gate is scheduled.
        """
        screen = self.screen
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

    def _start_font(self) -> dict[str, Font]:
        """Start the ``pygame`` fonts for the gate and axis font.

        Returns:
            ``pygame`` fonts for the gate and axis font.
        """
        pygame.font.init()
        return {
            "gate": pygame.font.SysFont("Arial", 12),
            "axis": pygame.font.SysFont("Arial", 30),
        }
