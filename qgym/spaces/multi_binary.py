"""
A multi-binary space, i.e. an array of binary values. A sample gives a random binary
array.
"""

from typing import List, Optional, Sequence, Union

import gym.spaces
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


class MultiBinary(gym.spaces.MultiBinary):
    """
    Multi-binary action/observation space for use in RL environments.
    """

    def __init__(
        self,
        n: Union[np.ndarray, Sequence[int], int],
        rng: Optional[Generator] = None,
    ):
        """
        Initialize a multi-discrete space, i.e. multiple discrete intervals of given
        sizes.

        :param n: Number of elements in the space
        :param rng: Random number generator to be used in this space. If `None` a new
            one will be constructed.
        """

        super(MultiBinary, self).__init__(n)
        self._np_random = rng  # this overrides the default behaviour of the gym space

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Seed the rng of this space.

        :param seed: Seed for the rng
        """

        self._np_random = default_rng(seed)
        return [seed]

    def sample(self) -> NDArray:
        """
        :return: Random sampled element of this space.
        """

        return self.np_random.randint(2, size=self.n, dtype=self.dtype)
