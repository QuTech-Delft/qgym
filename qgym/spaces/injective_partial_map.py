"""
A space that represents a partial injective map from x elements to n bins, with n >= x.
"""

from typing import Any, Iterable, Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from qgym import Space


class InjectivePartialMap(Space[NDArray[np.int_]]):
    def __init__(
        self, domain_size: int, codomain_size: int, rng: Optional[Generator] = None
    ):
        """
        Initialize a space that models actions for a partial injective map from x elements to n bins, with n>=x.

        :param domain_size: Number of elements to map (x).
        :param codomain_size: Number of bins to map elements to (n).
        :param rng: Random number generator to be used in this space. If `None` a new one will be constructed.
        """
        super().__init__(rng)
        self._domain_size = domain_size
        self._codomain_size = codomain_size

    def sample(self) -> NDArray[np.int_]:
        """
        Sample a random value from this space.

        :return: Random value from this space.
        """
        num_mapped = self.rng.integers(0, self._domain_size, endpoint=True)
        num_unmapped = self._codomain_size - num_mapped
        mapped = self.rng.choice(
            range(1, self._domain_size + 1), size=num_mapped, replace=False
        )
        return self.rng.permutation([-1] * num_unmapped + mapped)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, Iterable):
            value = np.array(value)
            if value.dtype == np.int_:
                mapped = np.setdiff1d(value, [-1])
                return len(value) == self._codomain_size and len(mapped) == len(
                    np.unique(mapped)
                )
        return False

    def __str__(self) -> str:
        return f"InjectivePartialMap({self._domain_size}->{self._codomain_size})"

    def __eq__(self, other):
        return (
            isinstance(other, InjectivePartialMap)
            and self._domain_size == other._domain_size
            and self._codomain_size == other._codomain_size
        )
