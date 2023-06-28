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
    connection_graph: Optional[nx.Graph] = None,
    connection_graph_matrix: Optional[ArrayLike] = None,
    connection_grid_size: Optional[Gridspecs] = None,
) -> nx.Graph:
    """Parse the user input (given in ``__init__``) to create a connection graph.

    :param connection_graph: ``networkx.Graph`` representation of the QPU topology.
    :param connection_graph_matrix: Adjacency matrix representation of the QPU
        topology
    :param connection_grid_size: Size of the connection graph when the topology is a
        grid.
    :raise ValueError: When `connection_graph`, `connection_graph_matrix` and
        `connection_grid_size` are all None.
    :return: Connection graph as a ``networkx.Graph``.
    """
    if connection_graph is not None:
        if connection_graph_matrix is not None:
            msg = "Both 'connection_graph' and 'connection_graph_matrix' were given. "
            msg += "Using 'connection_graph'."
            warnings.warn(msg)
        if connection_grid_size is not None:
            msg = "Both 'connection_graph' and 'connection_grid_size' were given. "
            msg += "Using 'connection_graph'."
            warnings.warn(msg)

        check_graph_is_valid_topology(connection_graph, "connection_graph")

        # deepcopy the graphs for safety
        return deepcopy(connection_graph)

    if connection_graph_matrix is not None:
        if connection_grid_size is not None:
            msg = "Both 'connection_graph_matrix' and 'connection_grid_size' were "
            msg += "given. Using 'connection_graph_matrix'."
            warnings.warn(msg)
        connection_graph_matrix = check_adjacency_matrix(connection_graph_matrix)
        return nx.from_numpy_array(connection_graph_matrix)
    if connection_grid_size is not None:
        # Generate connection grid graph
        return nx.grid_graph(connection_grid_size)

    msg = "No valid arguments for instantiation a connection graph were given"
    raise ValueError(msg)
