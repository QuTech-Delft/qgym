"""
A Box space, i.e. a possibly open-ended interval in n dimensions.
"""
from typing import Generator, List, Optional, Sequence, SupportsFloat, Type, Union

import gym.spaces
import numpy as np
from numpy.random import default_rng


class Box(gym.spaces.Box):
    """
    An n-dimensional box space, i.e. collection of possibly open-ended intervals.
    """

    def __init__(
        self,
        low: Union[SupportsFloat, np.ndarray],
        high: Union[SupportsFloat, np.ndarray],
        shape: Optional[Sequence[int]] = None,
        dtype: Type = np.float32,
        rng: Optional[Generator] = None,
    ):
        """
        Initialize a Box space, i.e. a possibly open-ended interval in n dimensions.

        :param low: Either one lower bound for all intervals, or an array of the correct shape with unique lower bound
            for each interval.
        :param high: Either one upper bound for all intervals, or an array of the correct shape with unique upper bound
            for each interval.
        :param shape: Shape of this space.
        :param dtype: Type of the values in each interval.
        :param rng: Random number generator to be used in this space. If `None` a new one will be constructed.
        """
        super(Box, self).__init__(low, high, shape=shape, dtype=dtype)
        self._np_random = rng  # this overrides the default behaviour of the gym space

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Seed the rng of this space
        :param seed: Seed for the rng
        """
        self._np_random = default_rng(seed)
        return [seed]
