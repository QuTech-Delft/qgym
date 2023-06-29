"""This module contains the ``Box`` space, i.e., a possibly open-ended interval in $n$
dimensions.

Usage:
    >>> from qgym.spaces import Box
    >>> Box(low=0, high=20, shape=(2, 3), dtype=int)
    Box([[0 0 0]
     [0 0 0]], [[20 20 20]
     [20 20 20]], (2, 3), int32)

"""
from typing import Any, List, Optional, Sequence, SupportsFloat, Type, Union

import gymnasium.spaces
import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


class Box(gymnasium.spaces.Box):
    """An $n$-dimensional box space, i.e., collection of (possibly) open-ended
    intervals.
    """

    def __init__(
        self,
        low: Union[SupportsFloat, NDArray[Any]],
        high: Union[SupportsFloat, NDArray[Any]],
        shape: Optional[Sequence[int]] = None,
        dtype: Type[np.floating[Any]] | Type[np.integer[Any]] = np.float_,
        *,
        rng: Optional[Generator] = None,
    ) -> None:
        """Initialize a ``Box`` space, i.e., a possibly open-ended interval in $n$
        dimensions.

        :param low: Either one lower bound for all intervals, or an ``NDArray`` with the
            shape given in `shape` with unique lower bounds for each interval.
        :param high: Either one upper bound for all intervals, or an ``NDArray`` with
            the shape given in `shape` with unique upper bounds for each interval.
        :param shape: ``Tuple`` containing the shape of the ``Box`` space.
        :param dtype: Type of the values in each interval.
        :param rng: Random number generator to be used in this space, if ``None`` a new
            one will be constructed.
        """
        super().__init__(low, high, shape=shape, dtype=dtype)
        self._np_random = rng  # this overrides the default behaviour of the gym space

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        self._np_random = default_rng(seed)
        return [seed]
