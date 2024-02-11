"""This module contains generic input validation methods."""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

# pylint: disable=invalid-name


def check_real(  # pylint: disable=too-many-arguments
    x: Any,
    name: str,
    *,
    l_bound: float | None = None,
    u_bound: float | None = None,
    l_inclusive: bool = True,
    u_inclusive: bool = True,
) -> float:
    """Check if the variable `x` with name 'name' is a real number.

    Optionally, lower and upper bounds can also be checked.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in possible error
            messages.
        l_bound: Lower bound of `x`.
        u_bound: Upper bound of `x`.
        l_inclusive: If ``True`` the lower bound is inclusive, otherwise the lower bound
            is exclusive.
        u_inclusive: If ``True`` the upper bound is inclusive, otherwise the upper bound
            is exclusive.

    Raises:
        TypeError: If `x` is not a real number.
        ValueError: If `x` is outside the give bounds.

    Returns:
        Floating point representation of `x`.
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


def check_int(  # pylint: disable=too-many-arguments
    x: Any,
    name: str,
    *,
    l_bound: float | None = None,
    u_bound: float | None = None,
    l_inclusive: bool = True,
    u_inclusive: bool = True,
) -> int:
    """Check if the variable `x` with name 'name' is a real number.

    Optionally, lower and upper bounds can also be checked.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in possible error
            messages.
        l_bound: Lower bound of `x`.
        u_bound: Upper bound of `x`.
        l_inclusive: If ``True`` the lower bound is inclusive, otherwise the lower
            bound is exclusive.
        u_inclusive: If ``True`` the upper bound is inclusive, otherwise the upper
            bound is exclusive.

    Raises:
        TypeError: If `x` is not a real number.
        ValueError: If `x` is outside the give bounds.

    Returns:
        Floating point representation of `x`.
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
    """Check if the variable `x` with name 'name' is a string.

    Optionally, the string can be converted to all lowercase or all uppercase letters.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in possible error
            messages.
        lower: If ``True``, `x` will be returned with lowercase letters. Defaults to
            ``False``.
        upper: If ``True``, `x` will be returned with uppercase letters. Default to
            ``False``.

    Raises:
        TypeError: If `x` is not an instance of ``str``.

    Returns:
        Input string. Optionally, in lowercase or uppercase letters.
    """
    if not isinstance(x, str):
        raise TypeError(f"'{name}' must be a string, but was of type {type(x)}")

    if lower:
        x = x.lower()
    if upper:
        x = x.upper()
    return x


def check_bool(x: Any, name: str, *, safe: bool = False) -> bool:
    """Check if the variable `x` with name 'name' is a Boolean value.

    Optionally, cast to Boolean value if it is not.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in possible error
            messages.
        safe: If ``True`` raise a ``TypeError`` when `x` is not a bool. If ``False``
            cast to bool.

    Raises:
        TypeError: If `safe` is ``True`` and `x` is not a Boolean value.

    Returns:
        Boolean representation of the input.
    """
    if not isinstance(x, bool) and safe:
        raise TypeError(f"'{name}' must be a Boolean value, but was of type {type(x)}")
    return bool(x)


def check_adjacency_matrix(adjacency_matrix: ArrayLike) -> NDArray[Any]:
    """Check if a matrix is an adjacency matrix, i.e., a square matrix.

    Args:
        adjacency_matrix: Matrix to check.

    Raises:
        ValueError: When the provided input is not a square matrix.

    Returns:
        Square NDArray representation of ``adjacency_matrix``.
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
    """Check if `graph` with name 'name' is an instance of ``networkx.Graph``, check
    if the graph is valid topology graph and check if the nodes are integers.

    Args:
        graph: Graph to check.
        name: Name of the graph. This name will be displayed in possible error
            messages.

    Raises:
        TypeError: If `graph` is not an instance of ``networkx.Graph``.
        ValueError: If `graph` is not a valid topology graph.
    """
    check_instance(graph, name, nx.Graph)

    if nx.number_of_selfloops(graph) > 0:
        raise ValueError(f"'{name}' contains self-loops")

    if len(graph) == 0:
        raise ValueError(f"'{name}' has no nodes")

    if not all(isinstance(node, int) for node in graph.nodes()):
        raise TypeError(f"'{name}' has nodes that are not integers")


def check_instance(x: Any, name: str, dtype: type) -> None:
    """Check if `x` with name 'name' is an instance of dtype.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in possible error
            messages.

    Raises:
        TypeError: If `x` is not an instance of `dtype`.
    """
    if not isinstance(x, dtype):
        msg = f"'{name}' must an instance of {dtype}, but was of type {type(x)}"
        raise TypeError(msg)


def warn_if_positive(x: float, name: str) -> None:
    """Give a warning when `x` is positive.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in the warning.
    """
    if x > 0:
        warnings.warn(f"'{name}' was positive")


def warn_if_negative(x: float, name: str) -> None:
    """Give a warning when `x` is negative.

    Args:
        x: Variable to check.
        name: Name of the variable. This name will be displayed in the warning.
    """
    if x < 0:
        warnings.warn(f"'{name}' was negative")
