"""This module contains types and type aliases used during visualisation."""

from typing import Tuple

import pygame

Color = Tuple[int, int, int]
Font = pygame.font.Font
Surface = pygame.surface.Surface

__all__ = ["Color", "Font", "Surface"]
