"""
A multi-discrete space, i.e. multiple discrete intervals. A sample takes one item from each interval.
"""

from typing import Any, Iterable, List, Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from qgym import Space


class MultiDiscrete(Space[NDArray[np.int_]]):
    def __init__(
        self,
        sizes: List[int],
        starts: Optional[List[int]] = None,
        rng: Optional[Generator] = None,
    ):
        """
        Initialize a multi-discrete space, i.e. multiple discrete intervals of given sizes and with given lowest values.

        :param sizes: Sizes of all intervals in order.
        :param starts: Start values of all intervals in order. If `None` each interval will start at 0.
        :param rng: Random number generator to be used in this space. If `None` a new one will be constructed.
        """
        super().__init__(rng)
        if starts is None:
            starts = np.zeros_like(sizes)
        if len(starts) != len(sizes):
            raise ValueError("Both `sizes` and `starts` should have the same length.")
        self._sizes = np.array(sizes)
        self._starts = np.array(starts)

    def sample(self) -> NDArray[np.int_]:
        """
        Sample a random value from this space.

        :return: Random value from this space.
        """
        return self._starts + self.rng.integers(0, self._sizes)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Iterable):
            value = np.array(value)
            if value.dtype == np.int_:
                return np.all(self._starts <= value) and np.all(value < self._sizes)
        return False

    def __str__(self) -> str:
        return f"Discrete({list(self._sizes)})"

    def __eq__(self, other: Any):
        return isinstance(other, MultiDiscrete) and self._sizes == other._sizes
