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
        self, sizes: List[int], starts: List[int], rng: Optional[Generator] = None
    ):
        super().__init__(rng)
        self._sizes = np.array(sizes)
        self._starts = np.array(starts)

    def sample(self) -> NDArray[np.int_]:
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
