"""This module contains the ``Box`` space, i.e., a possibly open-ended interval in $n$
dimensions.

Usage:
    >>> from qgym.spaces import Box
    >>> Box(low=0, high=20, shape=(2, 3), dtype=int)
    Box([[0 0 0]
     [0 0 0]], [[20 20 20]
     [20 20 20]], (2, 3), int32)

"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any, SupportsFloat

import gymnasium.spaces
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike


class Box(gymnasium.spaces.Box):
    """An $n$-dimensional box space, i.e., collection of (possibly) open-ended
    intervals.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        low: SupportsFloat | ArrayLike,
        high: SupportsFloat | ArrayLike,
        shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float64,
        *,
        rng: Generator | None = None,
    ) -> None:
        """Initialize a ``Box`` space, i.e., a possibly open-ended interval in $n$
        dimensions.

        Args:
            low: Either one lower bound for all intervals, or an ``NDArray`` with the
                shape given in `shape` with unique lower bounds for each interval.
            high: Either one upper bound for all intervals, or an ``NDArray`` with the
                shape given in `shape` with unique upper bounds for each interval.
            shape: ``Tuple`` containing the shape of the ``Box`` space.
            dtype: Type of the values in each interval.
            rng: Random number generator to be used in this space, if ``None`` a new one
                will be constructed.
        """
        if not isinstance(low, Integral):
            low = low.__float__() if hasattr(low, "__float__") else np.asarray(low)
        if not isinstance(high, Integral):
            high = high.__float__() if hasattr(high, "__float__") else np.asarray(high)
        super().__init__(low, high, shape=shape, dtype=dtype)
        self._np_random = rng  # this overrides the default behaviour of the gym space
