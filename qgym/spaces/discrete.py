"""This module contains the ``Discrete`` space, i.e., a range of integers.

A sample returns a randomly generated number within the bounds of the Discrete space.

Usage:
    >>> from qgym.spaces import Discrete
    >>> Discrete(3)
    Discrete(3)

"""
from typing import List, Optional

import gymnasium.spaces
from numpy.random import Generator, default_rng


class Discrete(gymnasium.spaces.Discrete):
    """Discrete action/observation space for use in RL environments."""

    def __init__(
        self,
        n: int,
        start: int = 0,
        *,
        rng: Optional[Generator] = None,
    ) -> None:
        """Initialize a Discrete space,  i.e., a range of integers.

        :param n: The number of integer values in the Discrete space.
        :param start: The smallest element is the Discrete Space.
        :param rng: Random number generator to be used in this space, if ``None`` a new
            random number generator will be constructed.
        """
        super().__init__(n=n, start=start)
        self._np_random = rng  # this overrides the default behaviour of the gym space

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        self._np_random = default_rng(seed)
        return [seed]
