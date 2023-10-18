"""This module contains function which parse user input.

With parsing we mean that the user input is validated and transformed to a predictable
format. In this way, user can give different input formats, but internally we are 
assured that the data has the same format."""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, Type

import networkx as nx
from numpy.typing import ArrayLike

from qgym.templates import Rewarder, Visualiser
from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_graph_is_valid_topology,
    check_instance,
    check_string,
)

if TYPE_CHECKING:
    Gridspecs = list[int | Iterable[int]] | tuple[int | Iterable[int]]


def parse_rewarder(rewarder: Rewarder | None, default: Type[Rewarder]) -> Rewarder:
    """Parse a `rewarder` given by the user.

    Args:
        rewarder: ``Rewarder`` to use for the environment. If ``None``, then a new
            instance of the `default` rewarder will be returned.
        default: Type of the desired default rewarder to used when no rewarder is given.

    Returns:
        A deepcopy of the given `rewarder` or a new instance of type `default` if
        `rewarder` is ``None``.
    """
    if rewarder is None:
        return default()
    check_instance(rewarder, "rewarder", Rewarder)
    return deepcopy(rewarder)


def parse_visualiser(
    render_mode: str | None, vis_type: Type[Visualiser], args: list[Any]
) -> None | Visualiser:
    """Parse a `Visualiser` by the render mode.

    Args:
        render_mode: If ``None`` return ``None``. Otherwise return a ``Visualiser`` of
            type `vis_type` with optional arguments given in `args`.
        vis_type: Type of ``Visualiser`` to make if `render_mode` is not ``None``.
        args: Additional argument to give to the init of the ``Visualiser`` if
            `vis_type` is not ``None``.

    Returns:
        If `render_mode` is ``None`` return ``None``. Otherwise return a ``Visualiser``
        of type `vis_type` with optional arguments given in `args`.
    """
    if render_mode is None:
        return None

    render_mode = check_string(render_mode, "render_mode", lower=True)
    return vis_type(render_mode, *args)


def parse_connection_graph(
    graph: nx.Graph | None = None,
    matrix: ArrayLike | None = None,
    grid_size: Gridspecs | None = None,
) -> nx.Graph:
    """Parse the user input (given in ``__init__``) to create a connection graph.

    Args:
        graph: ``networkx.Graph`` representation of the QPU topology.
        matrix: Adjacency matrix representation of the QPU topology.
        grid_size: Size of the connection graph when the topology is a grid.

    Raises:
        ValueError: When `graph`, `matrix` and `grid_size` are all ``None``.

    Returns:
        Connection graph as a ``networkx.Graph``.
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
