from numbers import Real
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_real(
    x: Real,
    name: str,
    *,
    l_bound: Optional[float] = None,
    u_bound: Optional[float] = None,
    l_inclusive=True,
    u_inclusive=True,
) -> float:
    """
    Checks if the variable x with name 'name' is a real number. Optionally lower and
    upper bounds can also be checked.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :param l_bound: Lower bound of x.
    :param u_bound: Upper bound of x.
    :param l_inclusive: If true the lower bound is inclusive. Otherwise the lower bound
        is exclusive.
    :param u_inclusive: If true the upper bound is inclusive. Otherwise the upper bound
        is exclusive.
    :raise TypeError: If x is not a real number.
    :raise ValueError: If x is outside of the give bounds.
    :return: Floating point representation of x.
    """
    if not isinstance(x, Real):
        raise TypeError(f"{name} should be a real number, but was of type {type(x)}.")

    if l_bound is not None:
        if l_inclusive:
            if x < l_bound:
                raise ValueError(
                    f"{name} has an inclusive lower bound of {l_bound}, but was {x}."
                )
        else:
            if x <= l_bound:
                raise ValueError(
                    f"{name} has an exclusive lower bound of {l_bound}, but was {x}."
                )

    if u_bound is not None:
        if u_inclusive:
            if x > u_bound:
                raise ValueError(
                    f"{name} has an inclusive upper bound of {u_bound}, but was {x}."
                )
        else:
            if x >= u_bound:
                raise ValueError(
                    f"{name} has an exclusive upper bound of {u_bound}, but was {x}."
                )

    return float(x)


def check_string(x: str, name: str, *, lower: bool = False, upper: bool = False) -> str:
    """
    Checks if the variable x with name 'name' is a string. Optionally the string can
    be converted to all lowercase or all uppercase letters.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :param lower: If True, x will be returned with lowercase letters. Default is False.
    :param upper: If True, x will be returned with uppercase letters. Default is False.
    :raise TypeError: If x is not of type str.
    :return: input string, optionally in lowercase or uppercase letters.
    """
    if not isinstance(x, str):
        raise TypeError(f"{name} must be a string, but was of type {type(x)}.")

    if lower:
        x = x.lower()
    if upper:
        x = x.upper()
    return x


def check_adjacency_matrix(adjacency_matrix: ArrayLike) -> NDArray[Any]:
    """
    Checks if a matrix is an adjacency matrix, i.e., a square matrix.

    :param adjacency_matrix: Matrix to check.
    :raise ValueError: When the provided input is not a valid matrix.
    """

    if hasattr(adjacency_matrix, "toarray"):
        adjacency_matrix = adjacency_matrix.toarray()
    else:
        adjacency_matrix = np.array(adjacency_matrix)

    if (
        not adjacency_matrix.ndim == 2
        and adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    ):
        raise ValueError("The provided value should be a square 2-D adjacency matrix.")

    return adjacency_matrix
