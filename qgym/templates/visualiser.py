"""Generic abstract base class for visualisers of RL environments.

All visualisers should inherit from ``Visualiser``.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import pygame
from numpy.typing import NDArray

from qgym.utils.visualisation.typing import Color, Font, Surface


class RenderData:
    """Class containing usefull data for rendering like screen, font and colors."""

    __slots__ = ["screen", "font", "colors", "render_mode"]

    def __init__(
        self,
        screen: Surface,
        font: dict[str, Font],
        colors: dict[str, Color],
        render_mode: str,
    ) -> None:
        self.screen = screen
        self.font = font
        self.colors = colors
        self.render_mode = render_mode

    @property
    def screen_width(self) -> int:
        """Screen width. Alias for ``self.screen_dimensions[0]``."""
        return self.screen.get_width()

    @property
    def screen_height(self) -> int:
        """Screen height. Alias for ``self.screen_dimensions[1]``."""
        return self.screen.get_height()


class Visualiser:
    """Visualizer for the the current state of the problem."""

    # --- These attributes should be set in any subclass ---
    render_data: RenderData

    @abstractmethod
    def __init__(self, render_mode: str, *args: list[Any]):
        raise NotImplementedError

    @abstractmethod
    def render(self, state: Any) -> None | NDArray[np.int_]:
        """Render the current state using ``pygame``."""
        raise NotImplementedError

    def step(self, state: Any) -> None:
        """To be used during a step of the environment.

        Renders the display if `render_mode` is 'human', does nothing otherwise.

        Args:
            state: State to render if `render_mode` is 'human'.
        """
        if self.render_data.render_mode == "human":
            self.render(state)

    def _display(self) -> None | NDArray[np.int_]:
        """Display the current state using ``pygame``.

        The render function should call this method at the end.

        Raises:
            ValueError: When an invalid mode is provided.

        Returns:
            If 'human' mode returns a boolean value encoding whether the ``pygame``
            screen is open. In 'rgb_array' mode returns an RGB array encoding of the
            rendered image.
        """
        if self.render_data.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            return None

        if self.render_data.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        msg = f"You provided an invalid mode '{self.render_data.render_mode}', the only"
        msg += " supported modes are 'human' and 'rgb_array'."
        raise ValueError(msg)

    def _start_screen(
        self, screen_name: str, render_mode: str, screen_dimensions: tuple[int, int]
    ) -> Surface:
        """Start a pygame screen in the given mode.

        Args:
            screen_name: Name of the screen.
            render_mode: The render mode to use. Choose from 'human' or 'rgb_array'.
            screen_dimension: Width and height of the screen.

        Raises:
            ValueError: When an invalid mode is provided.

        Returns:
            The initialized screen.
        """
        if not isinstance(render_mode, str):
            raise TypeError(
                f"'rendermode' of type {type(render_mode)} has no screen to start"
            )

        pygame.display.init()
        if render_mode == "human":
            screen = pygame.display.set_mode(screen_dimensions)
        elif render_mode == "rgb_array":
            screen = pygame.Surface(screen_dimensions)
        else:
            raise ValueError(
                f"You provided an invalid render mode '{render_mode}', the only "
                "supported modes are 'human' and 'rgb_array'."
            )
        pygame.display.set_caption(screen_name)
        return screen

    def close(self) -> None:
        """Close the screen used for rendering."""
        pygame.quit()

    @property
    def screen_width(self) -> int:
        """Screen width of  the main screen."""
        return self.render_data.screen_width

    @property
    def screen_height(self) -> int:
        """Screen height of the main screen."""
        return self.render_data.screen_height

    @property
    def is_open(self) -> bool:
        """Boolean value stating whether a ``pygame.screen`` is currently open."""
        return self.render_data.screen is not None

    @property
    def colors(self) -> dict[str, Color]:
        """Dict containing names color pairs."""
        return self.render_data.colors

    @property
    def screen(self) -> Surface:
        """Main screen to draw on."""
        return self.render_data.screen

    @property
    def font(self) -> dict[str, Font]:
        """Dict containing str Font pairs."""
        return self.render_data.font
