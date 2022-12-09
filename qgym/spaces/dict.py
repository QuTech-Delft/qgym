"""This module contains the ``Dict`` space, i.e., a dictionary with fixed strings as
keys and spaces as values.

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
import typing
from typing import List, Optional

import gym.spaces
from numpy.random import Generator, default_rng


class Dict(gym.spaces.Dict):
    """Dictionary of other action/observation spaces for use in RL environments."""

    def __init__(
        self,
        spaces: Optional[typing.Dict[str, gym.Space]] = None,
        *,
        rng: Optional[Generator] = None,
        **spaces_kwargs: gym.Space
    ) -> None:
        """Initialize a ``Dict`` space, with string valued keys and spaces inheriting
        from ``gym.Space`` as values.

        :param spaces: Dictionary containing string valued keys and spaces that are to
            form this ``Dict`` space.
        :param rng: Random number generator to be used in this space, if ``None`` a new
            one will be constructed.
        :param spaces_kwargs: Spaces that are to form this ``Dict`` space.
        """
        super().__init__(spaces, **spaces_kwargs)
        for space in self.spaces.values():
            # override the default behaviour of the gym space
            space._np_random = rng

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the rng of this space, using ``numpy.random.default_rng``. The seed will
        be applied to all spaces in the ``Dict`` space.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        for space in self.spaces.values():
            space._np_random = default_rng(seed)
        return [seed]
