"""This module contains the :class:`BenchmarkResult` class."""

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


class BenchmarkResult:

    def __init__(self, data: ArrayLike) -> None:
        """Init of the :class:`BenchmarkResult`.

        Args:
            data: 2-D ArrayLike containing benchmark data.
        """
        self._data = np.asarray(data)

    @property
    def raw_data(self) -> NDArray[Any]:
        """2DArray containing benchmark data."""
        return self._data

    def get_quartiles(self) -> NDArray[Any]:
        """Compute the quartiles for each metric.

        Returns:
            Array with shape (5, n_metrics) containing the quartiles for each metric.
        """
        return cast(
            NDArray[Any], np.quantile(self._data, [0, 0.25, 0.5, 0.75, 1], axis=1)
        )

    def get_median(self) -> NDArray[Any]:
        """Compute the median for each metric.

        Returns:
            Array with shape (n_metrics,) containing the median value for each metric.
        """
        return cast(NDArray[Any], np.median(self._data, axis=1))

    def get_mean(self) -> NDArray[Any]:
        """Compute the mean for each metric.

        Returns:
            Array with shape (n_metrics,) containing the mean value for each metric.
        """
        return cast(NDArray[Any], np.mean(self._data, axis=1))

    def __array__(self, dtype: DTypeLike = None, copy: bool = True) -> NDArray[Any]:
        """Convert the :class:`BenchmarkResult` data to an array."""
        return np.array(self._data, dtype=dtype, copy=copy)

    def __repr__(self) -> str:
        """String representation of the :class:`BenchmarkResult`."""
        return f"{self.__class__.__name__}{self._data}"

    def __eq__(self, other: Any) -> bool:
        """Check if this :class:`BenchmarkResult` is equal to `other`.

        Args:
            Other: object to compare to.

        Returns:
            Boolean value stating wether `self` is equal to `other`.
        """
        if not isinstance(other, BenchmarkResult):
            return False
        return np.array_equal(self._data, other._data)
