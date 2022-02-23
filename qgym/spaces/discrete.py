"""
A discrete space, i.e. a discrete interval.
"""
from typing import Any, Optional, SupportsInt

from numpy.random import Generator

from qgym import Space


class Discrete(Space[int]):
    def __init__(self, size: int, low: int = 0, rng: Optional[Generator] = None):
        super().__init__(rng)
        self._size = size
        self._start = low

    def sample(self) -> int:
        return self._start + self.rng.integers(0, self._size)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, SupportsInt):
            return self._start <= int(value) < self._start + self._size
        return False

    def __str__(self) -> str:
        if self._size == 2:
            return f"{{{self._start},{self._start+1}}}"
        if self._size == 3:
            return f"{{{self._start},{self._start + 1},{self._start + 2}}}"
        return f"{{{self._start},{self._start + 1},...,{self._start + self._size - 1}}}"

    def __eq__(self, other: Any):
        return (
            isinstance(other, Discrete)
            and self._size == other._size
            and self._start == other._start
        )
