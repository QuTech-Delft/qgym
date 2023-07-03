"""This module contains the ``MultiBinary`` space, i.e., an array of binary values.

Usage:
    >>> from qgym.spaces import MultiBinary
    >>> MultiBinary(10)
    MultiBinary(10)

"""
from typing import Optional, Sequence, Union

import gymnasium.spaces
import numpy as np
from numpy.random import Generator
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
