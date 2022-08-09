"""
A multi-discrete space, i.e. multiple discrete intervals. A sample takes one item
from each interval.
"""

from typing import List, Optional, Type, Union

import gym.spaces
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


class MultiDiscrete(gym.spaces.MultiDiscrete):
    """
    Multi-discrete action/observation space for use in RL environments.
    """

    def __init__(
        self,
        nvec: Union[List[int], NDArray[int]],
        dtype: Optional[Union[Type, str]] = np.int64,
        rng: Optional[Generator] = None,
    ):
        """
        Initialize a multi-discrete space, i.e. multiple discrete intervals of given
        sizes.

        :param nvec: Vector containing the size of each discrete interval.
        :param dtype: Type of the values in each interval (default np.int64)
        :param rng: Random number generator to be used in this space. If `None` a new
            one will be constructed.
        """

        super(MultiDiscrete, self).__init__(nvec, dtype=dtype)
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

        return (self.np_random.random(self.nvec.shape) * self.nvec).astype(self.dtype)
