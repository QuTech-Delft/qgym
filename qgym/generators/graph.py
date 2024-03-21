"""This module contains graph generators for :class:`~qgym.envs.InitialMapping`."""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterator, SupportsFloat, SupportsInt

import networkx as nx
from numpy.random import Generator

from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_int, check_real


class GraphGenerator(Iterator[nx.Graph]):  # pylint: disable=too-few-public-methods
    """Abstract Base Class for graph generation.

    All graph generators should inherit from :class:`GraphGenerator` to be compatible
    with the :class:`~qgym.envs.InitialMapping` environment.
    """

    finite: bool
    """Boolean value stating whether the generator is finite."""

    @abstractmethod
    def __next__(self) -> nx.Graph:
        """Make a new networkx ``Graph``.

        The __next__ method of a :class:`GraphGenerator` should generate a networkx
        ``Graph`` representation of the interaction graph. To be a valid interaction
        graph, all nodes should have integer labels starting from 0 and up to the number
        of nodes minus 1.
        """


class BasicGraphGenerator(GraphGenerator):
    """:class:`BasicGraphGenerator` is a simple graph generation implementation.

    It uses ``networkx`` `fast_gnp_random_graph`_ to generate graphs.

    .. _fast_gnp_random_graph: https://networkx.org/documentation/stable/reference/
       generated/networkx.generators.random_graphs.fast_gnp_random_graph.html
    """

    def __init__(
        self,
        n_nodes: SupportsInt,
        interaction_graph_edge_probability: SupportsFloat = 0.5,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`BasicGraphGenerator`.

        Args:
            n_nodes: Number of nodes in the generated graph.
            interaction_graph_edge_probability: Probability to add an edge between two
                nodes.
            seed: Seed to use.
        """
        self.n_nodes = check_int(n_nodes, "n_nodes", l_bound=1)
        self.interaction_graph_edge_probability = check_real(
            interaction_graph_edge_probability,
            "interaction_graph_edge_probability",
            l_bound=0,
            u_bound=1,
        )
        self.rng = parse_seed(seed)
        self.finite = False

    def __repr__(self) -> str:
        """String representation of the :class:`BasicGraphGenerator`."""
        n_nodes = self.n_nodes
        interaction_graph_edge_probability = self.interaction_graph_edge_probability
        rng = self.rng
        finite = self.finite
        return (
            f"BasicGraphGenerator[n_nodes={n_nodes}, "
            f"interaction_graph_edge_probability={interaction_graph_edge_probability}, "
            f"rng={rng}, "
            f"finite={finite}]"
        )

    def __next__(self) -> nx.Graph:
        """Create a new randomly generated graph."""
        return nx.fast_gnp_random_graph(
            n=self.n_nodes, p=self.interaction_graph_edge_probability, seed=self.rng
        )


class NullGraphGenerator(GraphGenerator):
    """Generator class for generating empty graphs.

    Useful for unit testing.
    """

    def __init__(self) -> None:
        """Init of the :class:`NullGraphGenerator`"""
        self.finite = False

    def __next__(self) -> nx.Graph:
        """Create a new null graph."""
        return nx.null_graph()

    def __repr__(self) -> str:
        """String representation of the :class:`NullGraphGenerator`."""
        return f"NullGraphGenerator[finite={self.finite}]"
