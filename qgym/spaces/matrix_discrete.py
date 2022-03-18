from typing import Any, Iterable, Optional, Tuple

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from qgym import Space


class MatrixDiscrete(Space[NDArray[np.int_]]):
    def __init__(
        self,
        low: Optional[int],
        high: Optional[int],
        shape: Tuple[int, ...],
        rng: Optional[Generator] = None,
    ):
        """
        Initialize a matrix shaped space with discrete values, i.e. matrix with values from a (possibly open-ended)
        discrete interval. If the interval is bounded on both sides, samples will be taken from a rounded uniform
        distribution. If the interval is bounded on one side, samples will be taken from a (shifted) rounded exponential
        distribution. If the interval is unbounded on both sides, samples will be taken from a rounded standard normal
        distribution.

        :param low: Lower bound of the discrete interval (if `None` the interval has no lower bound).
        :param high: Upper bound of the discrete interval (if `None` the interval has no upper bound).
        :param shape: Shape of the matrix.
        :param rng: Random number generator to be used in this space. If `None` a new one will be constructed.
        """
        super().__init__(rng)
        self._shape = shape
        self._low = low
        self._high = high

    def sample(self) -> NDArray[np.int_]:
        """
        Sample a random value from this space.

        :return: Random value from this space.
        """
        if self._high is None and self._low is None:
            return np.round(self.rng.normal(size=self._shape)).astype(np.int_)
        if self._high is None:
            return self._low + np.round(self.rng.exponential(size=self._shape)).astype(
                np.int_
            )
        if self._low is None:
            return self._high - np.round(self.rng.exponential(size=self._shape)).astype(
                np.int_
            )
        return np.round(
            self.rng.uniform(self._low, self._high, size=self._shape)
        ).astype(np.int_)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Iterable):
            value = np.array(value)
            if value.dtype.kind == "i":
                return np.all(self._low <= value) and np.all(value <= self._high)
        return False

    def __str__(self) -> str:
        return f"Discrete {self._shape} Matrix with value in the range [{self._low}, {self._high}]."

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MatrixDiscrete)
            and self._shape == other._shape
            and self._low == other._low
            and self._high == other._high
        )
