import warnings
from numbers import Integral, Real
from typing import Any, Optional

import networkx as nx
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
        raise TypeError(f"'{name}' should be a real number, but was of type {type(x)}")

    error_msg = "'" + name + "' has an {} {} bound of {}, but was " + str(x)
    if l_bound is not None:
        if l_inclusive:
            if x < l_bound:
                raise ValueError(error_msg.format("inclusive", "lower", l_bound))
        else:
            if x <= l_bound:
                raise ValueError(error_msg.format("exclusive", "lower", l_bound))

    if u_bound is not None:
        if u_inclusive:
            if x > u_bound:
                raise ValueError(error_msg.format("inclusive", "upper", u_bound))
        else:
            if x >= u_bound:
                raise ValueError(error_msg.format("exclusive", "upper", u_bound))

    return float(x)


def check_int(
    x: Integral,
    name: str,
    *,
    l_bound: Optional[float] = None,
    u_bound: Optional[float] = None,
    l_inclusive=True,
    u_inclusive=True,
) -> int:
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
        raise TypeError(f"'{name}' should be an integer, but was of type {type(x)}")

    if not isinstance(x, Integral):
        int_x = int(x)
        if x - int_x != 0:
            msg = f"'{name}' with value {x} could not be safely converted to an integer"
            raise ValueError(msg)

    error_msg = "'" + name + "' has an {} {} bound of {}, but was " + str(x)
    if l_bound is not None:
        if l_inclusive:
            if x < l_bound:
                raise ValueError(error_msg.format("inclusive", "lower", l_bound))
        else:
            if x <= l_bound:
                raise ValueError(error_msg.format("exclusive", "lower", l_bound))

    if u_bound is not None:
        if u_inclusive:
            if x > u_bound:
                raise ValueError(error_msg.format("inclusive", "upper", u_bound))
        else:
            if x >= u_bound:
                raise ValueError(error_msg.format("exclusive", "upper", u_bound))

    return int(x)


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
        raise TypeError(f"'{name}' must be a string, but was of type {type(x)}")

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
        adjacency_matrix.ndim != 2
        or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]
    ):
        raise ValueError("The provided value should be a square 2-D adjacency matrix.")

    return adjacency_matrix


def check_graph_is_valid_topology(graph: nx.Graph, name: str) -> None:
    """
    Checks if the graph with name 'name' is an instance of networkx.Graph and checks
    if the graph is valid topology graph.

    :param grapg: Graph to check.
    :param name: Name of the graph. This name will be displayed in possible error
        messages.
    :raise TypeError: If graph is not an instance networkx.Graph.
    :raise ValueError: If graph is not a valid topology graph.
    """

    check_instance(graph, name, nx.Graph)

    if nx.number_of_selfloops(graph) > 0:
        raise ValueError(f"'{name}' contains selfloops")

    if len(graph) == 0:
        raise ValueError(f"'{name}' has no nodes")


def check_instance(x: Any, name: str, dtype: type) -> None:
    """
    Checks if x with name 'name' is an instance of dtype.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in possible error
        messages.
    :raise TypeError: If x is not an instance dtype.
    """
    if not isinstance(x, dtype):
        msg = f"'{name}' must an instance of {dtype}, but was of type {type(x)}"
        raise TypeError(msg)


def warn_if_positive(x: Real, name: str) -> None:
    """
    Gives a warning when x is postive.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in the warning.
    """
    if x > 0:
        warnings.warn(f"'{name}' was postive")


def warn_if_negative(x: Real, name: str) -> None:
    """
    Gives a warning when x is negative.

    :param x: Variable to check.
    :param name: Name of the variable. This name will be displayed in the warning.
    """
    if x < 0:
        warnings.warn(f"'{name}' was negative")
