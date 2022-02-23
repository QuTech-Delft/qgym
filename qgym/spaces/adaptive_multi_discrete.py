"""
A multi-discrete space that can be updated after an action has been done, such that the same actions can not be
executed more than once.

Implemented as multiple sets, where one sample from the space consists of one item from each set. Once an item has been
used in a sample it should be removed from that set.
"""

from typing import Any, Generator, Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray

from qgym import Space


class AdaptiveMultiDiscrete(Space[NDArray[np.int_]]):
    def __init__(
        self, sizes: List[int], starts: List[int], rng: Optional[Generator] = None
    ):
        super().__init__(rng)
        self._initial_sizes = sizes
        self._initial_starts = starts
        self._sets = [
            [start + _ for _ in range(size)] for start, size in zip(starts, sizes)
        ]

    def update(self, sample: NDArray[np.int_]):
        for index, value in enumerate(sample):
            self._sets[index].remove(value)

    def reset(self):
        self._sets = [
            [start + _ for _ in range(size)]
            for start, size in zip(self._initial_starts, self._initial_sizes)
        ]

    def sample(self) -> NDArray[np.int_]:
        sample = []
        for set_ in self._sets:
            sample.append(self.rng.choice(set_))
        return np.array(sample)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Iterable):
            value = np.array(value)
            if value.dtype == np.int_:
                if len(value) != len(self._sets):
                    return False
                for index, value in enumerate(value):
                    if value not in self._sets[index]:
                        return False
                return True
        return False

    def __str__(self):
        return f"AdaptiveMultiDiscrete({self._sets})"
