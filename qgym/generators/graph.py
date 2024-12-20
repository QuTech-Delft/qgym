"""This module contains graph generators for :class:`~qgym.envs.InitialMapping`."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, SupportsFloat, SupportsInt

import networkx as nx

from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_graph_is_valid_topology, check_real

if TYPE_CHECKING:
    from numpy.random import Generator


class GraphGenerator(Iterator[nx.Graph]):
    """Abstract Base Class for graph generation.

    All graph generators should inherit from :class:`GraphGenerator` to be compatible
    with the :class:`~qgym.envs.InitialMapping` environment.
    """

    finite: bool
    """Boolean value stating whether the generator is finite."""

    @abstractmethod
    def __next__(self) -> nx.Graph:
        """Make a new :class:~`networkx.Graph`, representing an interaction graph.

        The __next__ method of a :class:`GraphGenerator` should generate a
        :class:~`networkx.Graph` representation of the interaction graph. To be a valid
        interaction graph, all nodes should have integer labels starting from 0 and up
        to the number of nodes minus 1.
        """

    @abstractmethod
    def set_state_attributes(self, **kwargs: Any) -> None:
        """Set attributes that the state can receive.

        This method is called inside the mapping environment to receive information
        about the state. The same keywords as for the the init of the
        :class:`~qgym.envs.initial_mapping.InitialMappingState` are provided.
        """


class BasicGraphGenerator(GraphGenerator):
    """:class:`BasicGraphGenerator` is a simple graph generation implementation.

    It uses :func:`~networkx.generators.random_graphs.fast_gnp_random_graph` to generate
    graphs.
    """

    def __init__(
        self,
        interaction_graph_edge_probability: SupportsFloat = 0.5,
        seed: Generator | SupportsInt | None = None,
    ) -> None:
        """Init of the :class:`BasicGraphGenerator`.

        Args:
            interaction_graph_edge_probability: Probability to add an edge between two
                nodes. See the documentation of
                :func:`~networkx.generators.random_graphs.fast_gnp_random_graph` for
                more information.
            seed: Seed to use.
        """
        self.interaction_graph_edge_probability = check_real(
            interaction_graph_edge_probability,
            "interaction_graph_edge_probability",
            l_bound=0,
            u_bound=1,
        )
        self.rng = parse_seed(seed)
        self.finite = False
        self.n_nodes: int

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
        """Create a new randomly generated :class:~`networkx.Graph`."""
        return nx.fast_gnp_random_graph(
            n=self.n_nodes, p=self.interaction_graph_edge_probability, seed=self.rng
        )

    def set_state_attributes(
        self, *, connection_graph: nx.Graph | None = None, **kwargs: Any
    ) -> None:
        """Set the `n_qubits` attribute.

        Args:
            connection_graph: A :class:`~networkx.Graph` representation of the
                connection graph.
            kwargs: Additional keyword arguments. These are not used
        """
        connection_graph = check_graph_is_valid_topology(
            connection_graph, "connection_graph"
        )
        self.n_nodes = connection_graph.number_of_nodes()


class NullGraphGenerator(GraphGenerator):
    """Generator class for generating empty :class:`~netowrkx.Graphs`.

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

    def set_state_attributes(self, **kwargs: Any) -> None:
        """Receive state attributes, but do nothing with it.

        Args:
            kwargs: Keyword arguments.
        """
