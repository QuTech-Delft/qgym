"""This module contains generic input validation methods."""
import warnings
from numbers import Integral, Real
from typing import Any, Optional, TypeVar

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_real(
    x: Any,
    name: str,
    *,
    l_bound: Optional[float] = None,
    u_bound: Optional[float] = None,
    l_inclusive: bool = True,
    u_inclusive: bool = True,
) -> float:
    """Check if the variable `x` with name 'name' is a real number. Optionally, lower
    and upper bounds can also be checked.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :param l_bound: Lower bound of `x`.
    :param u_bound: Upper bound of `x`.
    :param l_inclusive: If ``True`` the lower bound is inclusive, otherwise the lower
        bound is exclusive.
    :param u_inclusive: If ``True`` the upper bound is inclusive, otherwise the upper
        bound is exclusive.
    :raise TypeError: If `x` is not a real number.
    :raise ValueError: If `x` is outside the give bounds.
    :return: Floating point representation of `x`.
    """
    if not isinstance(x, Real):
        raise TypeError(f"'{name}' should be a real number, but was of type {type(x)}")
    x_float = float(x)
    error_msg = "'" + name + "' has an {} {} bound of {}, but was " + str(x)
    if l_bound is not None:
        if l_inclusive:
            if x_float < l_bound:
                raise ValueError(error_msg.format("inclusive", "lower", l_bound))
        else:
            if x_float <= l_bound:
                raise ValueError(error_msg.format("exclusive", "lower", l_bound))

    if u_bound is not None:
        if u_inclusive:
            if x_float > u_bound:
                raise ValueError(error_msg.format("inclusive", "upper", u_bound))
        else:
            if x_float >= u_bound:
                raise ValueError(error_msg.format("exclusive", "upper", u_bound))

    return x_float


def check_int(
    x: Any,
    name: str,
    *,
    l_bound: Optional[float] = None,
    u_bound: Optional[float] = None,
    l_inclusive: bool = True,
    u_inclusive: bool = True,
) -> int:
    """Check if the variable `x` with name 'name' is a real number. Optionally, lower
    and upper bounds can also be checked.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :param l_bound: Lower bound of `x`.
    :param u_bound: Upper bound of `x`.
    :param l_inclusive: If ``True`` the lower bound is inclusive, otherwise the lower
        bound is exclusive.
    :param u_inclusive: If ``True`` the upper bound is inclusive, otherwise the upper
        bound is exclusive.
    :raise TypeError: If `x` is not a real number.
    :raise ValueError: If `x` is outside the give bounds.
    :return: Floating point representation of `x`.
    """
    if not isinstance(x, Real):
        raise TypeError(f"'{name}' should be an integer, but was of type {type(x)}")

    if not isinstance(x, Integral):
        int_x = int(x)
        if x - int_x != 0:
            msg = f"'{name}' with value {x} could not be safely converted to an integer"
            raise ValueError(msg)

    error_msg = "'" + name + "' has an {} {} bound of {}, but was " + str(x)
    x_int = int(x)
    if l_bound is not None:
        if l_inclusive:
            if x_int < l_bound:
                raise ValueError(error_msg.format("inclusive", "lower", l_bound))
        else:
            if x_int <= l_bound:
                raise ValueError(error_msg.format("exclusive", "lower", l_bound))

    if u_bound is not None:
        if u_inclusive:
            if x_int > u_bound:
                raise ValueError(error_msg.format("inclusive", "upper", u_bound))
        else:
            if x_int >= u_bound:
                raise ValueError(error_msg.format("exclusive", "upper", u_bound))

    return x_int


def check_string(x: str, name: str, *, lower: bool = False, upper: bool = False) -> str:
    """Check if the variable `x` with name 'name' is a string. Optionally, the string
    can be converted to all lowercase or all uppercase letters.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :param lower: If ``True``, `x` will be returned with lowercase letters. Defaults to
        ``False``.
    :param upper: If ``True``, `x` will be returned with uppercase letters. Default to
        ``False``.
    :raise TypeError: If `x` is not an instance of ``str``.
    :return: Input string. Optionally, in lowercase or uppercase letters.
    """
    if not isinstance(x, str):
        raise TypeError(f"'{name}' must be a string, but was of type {type(x)}")

    if lower:
        x = x.lower()
    if upper:
        x = x.upper()
    return x


def check_adjacency_matrix(adjacency_matrix: ArrayLike) -> NDArray[Any]:
    """Check if a matrix is an adjacency matrix, i.e., a square matrix.

    :param adjacency_matrix: Matrix to check.
    :raise ValueError: When the provided input is not a square matrix.
    :return: Square NDArray representation of ``adjacency_matrix``.
    """

    numpy_matrix: NDArray[Any]
    if hasattr(adjacency_matrix, "toarray"):
        numpy_matrix = adjacency_matrix.toarray()
    else:
        numpy_matrix = np.array(adjacency_matrix)

    if numpy_matrix.ndim != 2 or numpy_matrix.shape[0] != numpy_matrix.shape[1]:
        raise ValueError("The provided value should be a square 2-D adjacency matrix.")

    return numpy_matrix


def check_graph_is_valid_topology(graph: nx.Graph, name: str) -> None:
    """Check if `graph` with name 'name' is an instance of ``networkx.Graph`` and check
    if the graph is valid topology graph.

    :param graph: Graph to check.
    :param name: Name of the graph. This name will be displayed in possible error
        messages.
    :raise TypeError: If `graph` is not an instance of ``networkx.Graph``.
    :raise ValueError: If `graph` is not a valid topology graph.
    """
    check_instance(graph, name, nx.Graph)

    if nx.number_of_selfloops(graph) > 0:
        raise ValueError(f"'{name}' contains self-loops")

    if len(graph) == 0:
        raise ValueError(f"'{name}' has no nodes")


def check_instance(x: Any, name: str, dtype: type) -> None:
    """Check if `x` with name 'name' is an instance of dtype.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :raise TypeError: If `x` is not an instance of `dtype`.
    """
    if not isinstance(x, dtype):
        msg = f"'{name}' must an instance of {dtype}, but was of type {type(x)}"
        raise TypeError(msg)


def warn_if_positive(x: float, name: str) -> None:
    """Give a warning when `x` is positive.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in the warning.
    """
    if x > 0:
        warnings.warn(f"'{name}' was positive")


def warn_if_negative(x: Real, name: str) -> None:
    """Give a warning when `x` is negative.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in the warning.
    """
    if x < 0:
        warnings.warn(f"'{name}' was negative")
