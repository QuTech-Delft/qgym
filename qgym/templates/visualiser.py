"""Generic abstract base class for visualisers of RL environments.

All visualisers should inherit from ``Visualiser``.
"""
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pygame
from numpy.typing import NDArray

from qgym.utils.visualisation.typing import Font, Surface


class Visualiser:
    """Visualizer for the the current state of the problem.

    Each subclass should set at least the following attributes:

    :ivar screen: Pygame screen. Can be ``None`` if no screen is open.
    :ivar screen_dimensions: Tuple containing the width height of the screen.
    :ivar font: Dictionary with different pygame fonts.
    """

    # --- These attributes should be set in any subclass ---
    screen: Optional[Surface]
    screen_dimensions: Tuple[int, int]
    font: Dict[str, Font]
    render_mode: str

    @abstractmethod
    def render(self, state: Any) -> Optional[NDArray[np.int_]]:
        """Render the current state using ``pygame``."""
        raise NotImplementedError

    def step(self, state: Any) -> None:
        """To be used during a step of the environment.

        Renders the display if `render_mode` is 'human', does nothing otherwise."""
        if self.render_mode == "human":
            self.render(state)

    def _display(self) -> Optional[NDArray[np.int_]]:
        """Display the current state using ``pygame``.

        The render function should call this method at the end.

        :raise ValueError: When an invalid mode is provided.
        :return: If 'human' mode returns a boolean value encoding whether the ``pygame``
            screen is open. In 'rgb_array' mode returns an RGB array encoding of the
            rendered image.
        """
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            return None

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(
                    pygame.surfarray.pixels3d(cast(pygame.surface.Surface, self.screen))
                ),
                axes=(1, 0, 2),
            )

        msg = f"You provided an invalid mode '{self.render_mode}', the only supported "
        msg += "modes are 'human' and 'rgb_array'."
        raise ValueError(msg)

    def _start_screen(self, screen_name: str) -> Surface:
        """Start a pygame screen in the given mode.

        :param screen_name: Name of the screen.
        :raise ValueError: When an invalid mode is provided.
        :return: The initialized screen.
        """
        pygame.display.init()
        if self.render_mode == "human":
            screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        elif self.render_mode == "rgb_array":
            screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            raise ValueError(
                f"You provided an invalid render mode '{self.render_mode}',"
                f" the only supported modes are 'human' and 'rgb_array'."
            )
        pygame.display.set_caption(screen_name)
        return screen

    def close(self) -> None:
        """Close the screen used for rendering."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

        self.font: Dict[str, Font] = {}

    @property
    def screen_width(self) -> int:
        """Screen width. Alias for ``self.screen_dimensions[0]``."""
        return self.screen_dimensions[0]

    @property
    def screen_height(self) -> int:
        """Screen height. Alias for ``self.screen_dimensions[1]``."""
        return self.screen_dimensions[1]

    @property
    def is_open(self) -> bool:
        """Boolean value stating whether a ``pygame.screen`` is currently open."""
        return self.screen is not None
