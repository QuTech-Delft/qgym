"""This module contains test for the ``BenchmarkResult`` class."""

import numpy as np
import pytest
from numpy.typing import NDArray

from qgym.benchmarks import BenchmarkResult


@pytest.fixture(name="raw_data")
def raw_data_fixture() -> NDArray[np.float64]:
    return np.arange(30).reshape(3, 10).astype(np.float64)


@pytest.fixture(name="result")
def result_fixture(raw_data: NDArray[np.float64]) -> BenchmarkResult:
    return BenchmarkResult(raw_data)


def test_raw_data(raw_data: NDArray[np.float64], result: BenchmarkResult) -> None:
    assert raw_data is result.raw_data


def test_median(result: BenchmarkResult) -> None:
    np.testing.assert_array_equal([4.5, 14.5, 24.5], result.get_median())


def test_mean(result: BenchmarkResult) -> None:
    np.testing.assert_array_equal([4.5, 14.5, 24.5], result.get_mean())


def test_quartiles(result: BenchmarkResult) -> None:
    np.testing.assert_array_equal(
        [
            [0.0, 10.0, 20.0],
            [2.25, 12.25, 22.25],
            [4.5, 14.5, 24.5],
            [6.75, 16.75, 26.75],
            [9.0, 19.0, 29.0],
        ],
        result.get_quartiles(),
    )


def test_ineq(result: BenchmarkResult) -> None:
    assert result is not None


def test_eq(result: BenchmarkResult) -> None:
    other = BenchmarkResult([range(10), range(10, 20), range(20, 30)])
    assert result == other


def test_repr(result: BenchmarkResult) -> None:
    expected_string = (
        "BenchmarkResult[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]\n "
        "[10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]\n "
        "[20. 21. 22. 23. 24. 25. 26. 27. 28. 29.]]"
    )
    assert repr(result) == expected_string
