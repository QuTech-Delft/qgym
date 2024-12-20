"""This module contains wrappers around some commonly used ``pygame`` functions."""

import numpy as np
import pygame
from numpy.typing import ArrayLike
from pygame import gfxdraw

from qgym.utils.visualisation.typing import Color, Font, Surface

# pylint: disable=invalid-name


def draw_point(screen: Surface, pos: ArrayLike, color: Color, r: int = 10) -> None:
    """Draw a point on the screen.

    Args:
        screen: Screen to add the point to.
        pos: ``ArrayLike`` containing the x and y coordinates of the point. Non integer
            values will be rounded down to the nearest integer.
        r: Radius of the point (in pixels). Defaults to 10.
    """
    pos_x, pos_y = np.asarray(pos, dtype=int)
    gfxdraw.aacircle(screen, pos_x, pos_y, r, color)
    gfxdraw.filled_circle(screen, pos_x, pos_y, r, color)


def draw_wide_line(
    screen: Surface,
    color: Color,
    point1: ArrayLike,
    point2: ArrayLike,
    *,
    width: int = 2,
) -> None:
    """Draw a wide line on the screen.

    Args:
        screen: Screen to draw the line on.
        color: Color of the line.
        point1: Coordinates of the starting point of the line.
        point2: Coordinates of the end point of the line.
        width: Width of the line. Defaults to 2.
    """
    # distance between the points
    p1 = np.asarray(point1)
    p2 = np.asarray(point2)
    distance = np.linalg.norm(p2 - p1)

    # scaled perpendicular vector (vector from p1 & p2 to the polygon's points)
    sp = np.array([p1[1] - p2[1], p2[0] - p1[0]]) * 0.5 * width / distance

    # points
    points = (p1 - sp, p1 + sp, p2 + sp, p2 - sp)

    # draw the polygon
    pygame.gfxdraw.aapolygon(screen, points, color)  # type: ignore[arg-type]
    pygame.gfxdraw.filled_polygon(screen, points, color)  # type: ignore[arg-type]


def write_text(
    screen: Surface, font: Font, text: str, pos: ArrayLike, color: Color
) -> None:
    """Write text, centered around the given point.

    Args:
        screen: Screen to write the text on.
        font: Font to use.
        text: Text to write.
        pos: x and y coordinates of the center of the text.
        color: Color of the text.
    """
    pos_x, pos_y = np.asarray(pos, dtype=int)
    pygame_text = font.render(text, antialias=True, color=color)
    text_position = pygame_text.get_rect(center=(pos_x, pos_y))
    screen.blit(pygame_text, text_position)


def shade_rect(
    screen: Surface, size: ArrayLike, pos: ArrayLike, color: Color, alpha: int
) -> None:
    """Shade a rectangular area.

    Args:
        screen: Screen to draw the shade on.
        size: Width and height of the rectangular area.
        pos: The x,y-coordinates of the lower left corner of the rectangle.
        color: Color of the shaded area.
        alpha: Alpha of the shade.
    """
    rect_width, rect_height = np.asarray(size, dtype=int)
    pos_x, pos_y = np.asarray(pos, dtype=int)

    surf = pygame.Surface((rect_width, rect_height))
    surf.set_alpha(alpha)
    surf.fill(color)
    screen.blit(surf, (pos_x, pos_y))
