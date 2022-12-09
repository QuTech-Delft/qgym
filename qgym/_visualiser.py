"""Generic abstract base class for visualisers of RL environments. All visualisers
should inherit from ``Visualiser``.
"""
from abc import abstractmethod
from typing import Any, Optional, cast

import numpy as np
import pygame


class Visualiser:
    """Visualizer for the the current state of the problem.

    Each subclass should set at least the following attributes:

    :ivar screen: Pygame screen.
    :ivar is_open: Boolean variable that states if the screen is open.
    :ivar screen_width: Width of the screen.
    :ivar screen_height: Height of the screen.
    """

    # --- These attributes should be set in any subclass ---
    screen: Optional[pygame.surface.Surface]
    is_open: bool
    screen_width: int
    screen_height: int

    @abstractmethod
    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Render the current state using ``pygame``."""
        raise NotImplementedError

    def _display(self, mode: str) -> Any:
        """Display the current state using ``pygame``.

        The render function should call this method at the end.

        :param mode: Mode to start pygame for ("human" and "rgb_array" are supported).
        :raise ValueError: When an invalid mode is provided.
        :return: In 'human' mode returns a boolean value encoding whether the ``pygame``
            screen is open. In 'rgb_array' mode returns an RGB array encoding of the
            rendered image.
        """
        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            return self.is_open

        if mode == "rgb_array":
            return np.transpose(
                np.array(
                    pygame.surfarray.pixels3d(cast(pygame.surface.Surface, self.screen))
                ),
                axes=(1, 0, 2),
            )

        msg = f"You provided an invalid mode '{mode}', the only supported modes are "
        msg += "'human' and 'rgb_array'."
        raise ValueError(msg)

    def _start_screen(self, screen_name: str, mode: str) -> pygame.surface.Surface:
        """Start a pygame screen in the given mode.

        :param screen_name: Name of the screen.
        :param mode: Mode to start pygame for ("human" and "rgb_array" are supported).
        :raise ValueError: When an invalid mode is provided.
        :return: The initialized screen.
        """
        pygame.display.init()
        if mode == "human":
            screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        elif mode == "rgb_array":
            screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            raise ValueError(
                f"You provided an invalid mode '{mode}',"
                f" the only supported modes are 'human' and 'rgb_array'."
            )
        pygame.display.set_caption(screen_name)
        self.is_open = True
        return screen

    def close(self) -> None:
        """Close the screen used for rendering."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()
            self.is_open = False
            self.screen = None
