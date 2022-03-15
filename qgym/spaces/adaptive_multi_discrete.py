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
        self,
        sizes: List[int],
        starts: Optional[List[int]] = None,
        rng: Optional[Generator] = None,
    ):
        """
        Initialize an adaptive multi-discrete space, i.e. multiple discrete intervals of given sizes and with given
        lowest values. This state can be updated with samples, such that the state space removes all states that have
        (partial) overlap with the given sample.

        :param sizes: Sizes of all intervals in order.
        :param starts: Start values of all intervals in order. If `None` each interval will start at 0.
        :param rng: Random number generator to be used in this space. If `None` a new one will be constructed.
        """
        super().__init__(rng)
        if starts is None:
            starts = np.zeros_like(sizes)
        if len(starts) != len(sizes):
            raise ValueError("Both `sizes` and `starts` should have the same length.")
        self._initial_sizes = sizes
        self._initial_starts = starts
        self._sets = [
            [start + _ for _ in range(size)] for start, size in zip(starts, sizes)
        ]

    def update(self, sample: NDArray[np.int_]) -> None:
        """
        Update this space with the given sample. I.e. remove all states that have (partial) overlap with this sample.

        :param sample: Sample to update this space with.
        """
        for index, value in enumerate(sample):
            self._sets[index].remove(value)

    def reset(self) -> None:
        """
        Resset this space to its initial state. I.e. all states are allowed again.
        """
        self._sets = [
            [start + _ for _ in range(size)]
            for start, size in zip(self._initial_starts, self._initial_sizes)
        ]

    def sample(self) -> NDArray[np.int_]:
        """
        Sample a random value from this space.

        :return: Random value from this space.
        """
        sample = []
        for set_ in self._sets:
            sample.append(self.rng.choice(set_))
        return np.array(sample)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Iterable):
            value = np.array(value)
            if value.dtype.kind == "i":
                if len(value) != len(self._sets):
                    return False
                for index, value in enumerate(value):
                    if value not in self._sets[index]:
                        return False
                return True
        return False

    def __str__(self):
        return f"AdaptiveMultiDiscrete({self._sets})"
