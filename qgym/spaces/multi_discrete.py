"""This module contains the ``MultiDiscrete`` space.

A multi-discrete space is a collection of multiple discrete intervals of given sizes.

Usage:
    >>> from qgym.spaces import MultiDiscrete
    >>> MultiDiscrete(nvec=[2,3,4])
    MultiDiscrete([2 3 4])

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium.spaces
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import ArrayLike


class MultiDiscrete(gymnasium.spaces.MultiDiscrete):
    """Multi-discrete action/observation space for use in RL environments."""

    def __init__(
        self,
        nvec: ArrayLike,
        dtype: str | type[np.integer[Any]] = np.int_,
        *,
        rng: Generator | None = None,
    ) -> None:
        """Initialize a multi-discrete space.

        A multi-discrete space is a collection of multiple discrete intervals of given
        sizes.

        Args:
            nvec: Vector containing the upper bound of each discrete interval. The lower
                bound is always set to 0.
            dtype: Type of the values in each interval (default np.int64).
            rng: Random number generator to be used in this space, if ``None`` a new
                random number generator will be constructed.
        """
        super().__init__(nvec=np.asarray(nvec), dtype=dtype)
        self._np_random = rng  # this overrides the default behaviour of the gym space
