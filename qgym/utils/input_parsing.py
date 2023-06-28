"""This module contains function which parse user input.

With parsing we mean that the user input is validated and transformed to a predictable
format. In this way, user can give different input formats, but internally we are 
assured that the data has the same format."""
import warnings
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple, Type, Union

import networkx as nx
from numpy.typing import ArrayLike

from qgym.templates import Rewarder
from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_graph_is_valid_topology,
    check_instance,
)

Gridspecs = Union[List[Union[int, Iterable[int]]], Tuple[Union[int, Iterable[int]]]]


def parse_rewarder(rewarder: Optional[Rewarder], default: Type[Rewarder]) -> Rewarder:
    """Parse a `rewarder` given by the user.

    :param rewarder: ``Rewarder`` to use for the environment. If ``None``, then a new
        instance of the `default` rewarder will be returned.
    :param default: Type of the desired default rewarder to used when no rewarder is
        given.
    :return: A deepcopy of the given `rewarder` or a new instance of type `default` if
        `rewarder` is ``None``.
    """
    if rewarder is None:
        return default()
    check_instance(rewarder, "rewarder", Rewarder)
    return deepcopy(rewarder)


def parse_connection_graph(
    graph: Optional[nx.Graph] = None,
    matrix: Optional[ArrayLike] = None,
    grid_size: Optional[Gridspecs] = None,
) -> nx.Graph:
    """Parse the user input (given in ``__init__``) to create a connection graph.

    :param graph: ``networkx.Graph`` representation of the QPU topology.
    :param matrix: Adjacency matrix representation of the QPU topology.
    :param size: Size of the connection graph when the topology is a grid.
    :raise ValueError: When `graph`, `matrix` and `grid_size` are all ``None``.
    :return: Connection graph as a ``networkx.Graph``.
    """
    if graph is not None:
        if matrix is not None:
            warnings.warn("Both 'graph' and 'matrix' were given. Using 'graph'.")
        if grid_size is not None:
            warnings.warn("Both 'graph' and 'grid_size' were given. Using 'graph'.")

        check_graph_is_valid_topology(graph, "graph")

        # deepcopy the graphs for safety
        return deepcopy(graph)

    if matrix is not None:
        if grid_size is not None:
            warnings.warn("Both 'matrix' and 'grid_size' were given. Using 'matrix'.")
        matrix = check_adjacency_matrix(matrix)
        return nx.from_numpy_array(matrix)

    if grid_size is not None:
        # Generate connection grid graph
        return nx.grid_graph(grid_size)

    raise ValueError("No valid arguments for a connection graph were given")
