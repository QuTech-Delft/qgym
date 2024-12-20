"""This module contains function which parse user input.

With parsing we mean that the user input is validated and transformed to a predictable
format. In this way, users can give different input formats, but internally we are
assured that the data has the same format.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, SupportsInt

import networkx as nx
from numpy.random import Generator, default_rng

from qgym.templates import Rewarder, Visualiser
from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_graph_is_valid_topology,
    check_instance,
    check_string,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def parse_rewarder(rewarder: Rewarder | None, default: type[Rewarder]) -> Rewarder:
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
    render_mode: str | None, vis_type: type[Visualiser], args: list[Any]
) -> Visualiser | None:
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
    graph: nx.Graph | ArrayLike | list[int] | tuple[int, ...],
) -> nx.Graph:
    """Parse the user input (given in ``__init__``) to create a connection graph.

    Args:
        graph: A graph representation of a connection graph. Supported input formats are
            a graph represented as a :class:`~networkx.Graph`, an adjacency  matrix
            represented as an :class:`~numpy.ArrayLike` or gridsize specification.

    Returns:
        Connection graph as a :class:`~networkx.Graph`.
    """
    if isinstance(graph, nx.Graph):
        check_graph_is_valid_topology(graph, "graph")
        # deepcopy the graphs for safety
        return deepcopy(graph)

    if isinstance(graph, (list, tuple)) and isinstance(graph[0], int):
        grid_graph = nx.grid_graph(graph)
        # Relabel the nodes to be integers
        return nx.convert_node_labels_to_integers(grid_graph)
    try:
        adj_mat = check_adjacency_matrix(graph)
    except ValueError as err:
        msg = "no valid arguments for a connection graph were given"
        raise ValueError(msg) from err

    return nx.from_numpy_array(adj_mat)


def has_fidelity(graph: nx.Graph) -> bool:
    """Check if the given graph has fidelity encoded in its edges.

    A graph can have fidelity encoded by setting the "weight" attributes of at least one
    edge to a fidelity value.

    Args:
        graph: Networks representation of the graph.

    Returns:
        Boolean value stating wether the graph has fidelity encoded.
    """
    for _, _, weight in graph.edges.data("weight"):
        if weight is not None and weight not in (0, 1):
            return True
    return False


def parse_seed(seed: Generator | SupportsInt | None) -> Generator:
    """Parse a give seed to produce a numpy.random.Generator.

    Args:
        seed: Seed to parse.

    Returns:
        A generator created from the given `seed`.

    Raises:
        TypeError: If `seed` is not a ``Generator``, ``SupportsInt`` or ``None``.
    """
    if seed is None:
        return default_rng(None)

    if isinstance(seed, SupportsInt):
        return default_rng(int(seed))

    if isinstance(seed, Generator):
        return seed

    msg = f"seed must be a Generator, SupportsInt or None, but was of type {type(seed)}"
    raise TypeError(msg)
