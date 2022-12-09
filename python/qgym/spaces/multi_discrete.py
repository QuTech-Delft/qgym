"""This module contains the ``MultiDiscrete`` space, i.e., multiple discrete intervals.
A sample returns one item from each interval.

Usage:
    >>> from qgym.spaces import MultiDiscrete
    >>> MultiDiscrete(nvec=[2,3,4])
    MultiDiscrete([2 3 4])

"""
from typing import List, Optional, Union

import gym.spaces
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import DTypeLike, NDArray


class MultiDiscrete(gym.spaces.MultiDiscrete):
    """Multi-discrete action/observation space for use in RL environments."""

    def __init__(
        self,
        nvec: Union[List[int], NDArray[np.int_]],
        dtype: DTypeLike = np.int64,
        rng: Optional[Generator] = None,
    ) -> None:
        """Initialize a multi-discrete space, i.e., multiple discrete intervals of given
        sizes.

        :param nvec: Vector containing the upper bound of each discrete interval. The
            lower bound is always set to 0.
        :param dtype: Type of the values in each interval (default np.int64)
        :param rng: Random number generator to be used in this space, if ``None`` a new
            random number generator will be constructed.
        """
        super().__init__(nvec, dtype=dtype)
        self._np_random = rng  # this overrides the default behaviour of the gym space

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        self._np_random = default_rng(seed)
        return [seed]

    def sample(self) -> NDArray[np.int_]:
        """Sample a random element from this space.

        :return: ``NDArray`` of shape (nvec,) containing random values from each
            discrete space.
        """
        sample = self.np_random.random(self.nvec.shape) * self.nvec
        return np.array(sample, dtype=self.dtype)
