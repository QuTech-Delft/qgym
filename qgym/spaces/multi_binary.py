"""This module contains the ``MultiBinary`` space, i.e., an array of binary values.

Usage:
    >>> from qgym.spaces import MultiBinary
    >>> MultiBinary(10)
    MultiBinary(10)

"""
from typing import List, Optional, Sequence, Union

import gymnasium.spaces
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


class MultiBinary(gymnasium.spaces.MultiBinary):
    """Multi-binary action/observation space for use in RL environments."""

    def __init__(
        self,
        n: Union[NDArray[np.int_], Sequence[int], int],
        *,
        rng: Optional[Generator] = None,
    ) -> None:
        """Initialize a multi-discrete space, i.e., multiple discrete intervals of given
        sizes.

        :param n: Number of elements in the space.
        :param rng: Random number generator to be used in this space. If ``None``, a new
            random number generator will be constructed.
        """
        super().__init__(n)
        self._np_random = rng  # this overrides the default behaviour of the gym space

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        self._np_random = default_rng(seed)
        return [seed]
