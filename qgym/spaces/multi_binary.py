"""This module contains the ``MultiBinary`` space, i.e., an array of binary values.

Usage:
    >>> from qgym.spaces import MultiBinary
    >>> MultiBinary(10)
    MultiBinary(10)

"""

from __future__ import annotations

import gymnasium.spaces
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike


class MultiBinary(gymnasium.spaces.MultiBinary):
    """Multi-binary action/observation space for use in RL environments."""

    def __init__(
        self,
        n: ArrayLike | int,
        *,
        rng: Generator | None = None,
    ) -> None:
        """Initialize a multi-discrete space, i.e., multiple discrete intervals of given
        sizes.

        Args:
            n: ArrayLike containing integers representing the number of elements in the
                space.
            rng: Random number generator to be used in this space. If ``None``, a new
                random number generator will be constructed.
        """
        if isinstance(n, int):
            super().__init__(n)
        else:
            super().__init__(np.asarray(n))
        self._np_random = rng  # this overrides the default behaviour of the gym space
