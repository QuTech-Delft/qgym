"""This module contains the ``Dict`` space.

The ``Dict`` space is a dictionary with fixed strings as keys and spaces as values.

Usage:
    >>> from qgym.spaces import Box, Dict
    >>> box1 = Box(low=0, high=20, shape=(2, 3), dtype=int)
    >>> box2 = Box(low=-5, high=1, shape=(3, 2), dtype=float)
    >>> Dict(box1=box1, box2=box2)
    Dict(box1:Box([[0 0 0]
     [0 0 0]], [[20 20 20]
     [20 20 20]], (2, 3), int32), box2:Box([[-5. -5.]
     [-5. -5.]
     [-5. -5.]], [[1. 1.]
     [1. 1.]
     [1. 1.]], (3, 2), float64))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium.spaces

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator


class Dict(gymnasium.spaces.Dict):
    """Dictionary of other action/observation spaces for use in RL environments."""

    def __init__(
        self,
        spaces: (
            dict[str, gymnasium.Space[Any]]
            | Sequence[tuple[str, gymnasium.Space[Any]]]
            | None
        ) = None,
        *,
        rng: Generator | None = None,
        **spaces_kwargs: gymnasium.Space[Any],
    ) -> None:
        """Initialize a ``Dict`` space.

        The string valued keys and spaces inheriting from ``gym.Space`` as values.

        Args:
            spaces: Dictionary containing string valued keys and spaces that are to form
            this ``Dict`` space.
            rng: Random number generator to be used in this space, if ``None`` a new one
                will be constructed.
            spaces_kwargs: Spaces that are to form this ``Dict`` space.
        """
        super().__init__(spaces, seed=None, **spaces_kwargs)
        for space in self.spaces.values():
            # override the default behaviour of the gym space
            space._np_random = rng  # noqa: SLF001
