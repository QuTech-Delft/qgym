r"""This module contains an environment for training an RL agent on the initial mapping
problem of OpenQL. The initial mapping problem is aimed at mapping virtual qubits of a
circuit to physical qubits that have a certain connection topology. The quantum circuit
is represented as an **interaction graph**, where each node represent a qubit and each
edge represent an interaction between two qubits as defined by the circuit (See the
example below). The QPU structure is called the **connection graph**. In the connection
graph each node represents a physical qubit and each edge represent a connection between
two qubits in the QPU.


.. code-block:: console

              QUANTUM CIRCUIT                        INTERACTION GRAPH
           ┌───┐               ┌───┐
    |q3>───┤ R ├───┬───────────┤ M ╞══                 q1 ────── q2
           └───┘   │           └───┘                            ╱
           ┌───┐ ┌─┴─┐         ┌───┐                           ╱
    |q2>───┤ R ├─┤ X ├───┬─────┤ M ╞══                        ╱
           └───┘ └───┘   │     └───┘                         ╱
           ┌───┐       ┌─┴─┐   ┌───┐                        ╱
    |q1>───┤ R ├───┬───┤ X ├───┤ M ╞══                     ╱
           └───┘   │   └───┘   └───┘                      ╱
           ┌───┐ ┌─┴─┐         ┌───┐                     ╱
    |q0>───┤ R ├─┤ X ├─────────┤ M ╞══                q3 ─────── q4
           └───┘ └───┘         └───┘



The goal is to create a mapping between the nodes of the interaction and connection
graph, such that for every edge in the interaction graph, there is an edge in the
connection graph. If this is impossible, then the number of mismatches should be
penalized.


State Space:
    The state space is described by a
    :class:`~qgym.envs.initial_mapping.InitialMappingState` with the following
    attributes:

    * `steps_done`: Number of steps done since the last reset.
    * `n_nodes`: Number of *physical* qubits.
    * `graphs`: Dictionary containing the graph and matrix representations of the both
      the interaction graph and connection graph.
    * `mapping`: Array of which the index represents a physical qubit, and the value a
      virtual qubit. A value of ``n_nodes + 1`` represents the case when nothing is
      mapped to the physical qubit yet.
    * `mapping_dict`: Dictionary that maps logical qubits (keys) to physical qubit
      (values).
    * `mapped_qubits`: Dictionary with a two Sets containing all mapped physical and
      logical qubits.

Observation Space:
    The observation space is a :class:`~qgym.spaces.Dict` with 2 entries:

    * `mapping`: The current state of the mapping.
    * `interaction_matrix`: The flattened adjacency matrix of the interaction graph.

Action Space:
    A valid action is a tuple of integers  $(i,j)$, such that  $0 \le i, j < n$, where
    $n$ is the number of physical qubits. The action  $(i,j)$ maps virtual qubit $j$ to
    phyiscal qubit $i$ when this action is legal. An action is legal when:

    #. virtual qubit $i$ has not been mapped to another physical qubit; and
    #. no other virual qubit has been mapped to physical qubit $j$.

Example 1:
    Creating an environment with a gridlike connection graph is done by executing the
    following code:

    >>> from qgym.envs.initial_mapping import InitialMapping
    >>> env = InitialMapping(0.5, connection_grid_size=(3,3))

    By default,  :class:`~qgym.envs.InitialMapping` uses the
    :class:`~qgym.envs.initial_mapping.BasicRewarder`. As an example, we would like to
    change the rewarder to the :class:`~qgym.envs.initial_mapping.EpisodeRewarder`. This
    can be done in the following way:

    >>> from qgym.envs.initial_mapping import EpisodeRewarder
    >>> env.rewarder = EpisodeRewarder()


Example 2:
    In this example we use a custom connection graph depicted in the code block below.

    .. code-block:: console

        q1──────q0──────q2
                 │
                 │
                 │
                q3


    The graph has a non-gridlike structure.
    Such connection graphs can be given to the environment by giving an adjacency matrix
    representation of the graph, or a ``networkx`` representation of the graph. We will
    show the latter option in this example.

    .. code-block:: python

        import networkx as nx
        from qgym.envs.initial_mapping import InitialMapping

        # Create a networkx representation of the connection graph
        connection_graph = nx.Graph()
        connection_graph.add_edge(0, 1)
        connection_graph.add_edge(0, 2)
        connection_graph.add_edge(0, 3)

        # Initialize the environment with the custom connection graph
        env = InitialMapping(0.5, connection_graph=connection_graph)


"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

import qgym.spaces
from qgym.envs.initial_mapping.graph_generation import (
    BasicGraphGenerator,
    GraphGenerator,
)
from qgym.envs.initial_mapping.initial_mapping_rewarders import BasicRewarder
from qgym.envs.initial_mapping.initial_mapping_state import InitialMappingState
from qgym.envs.initial_mapping.initial_mapping_visualiser import (
    InitialMappingVisualiser,
)
from qgym.templates import Environment, Rewarder
from qgym.utils.input_parsing import (
    parse_connection_graph,
    parse_rewarder,
    parse_visualiser,
)
from qgym.utils.input_validation import check_instance

if TYPE_CHECKING:
    Gridspecs = (
        list[int] | list[Iterable[int]] | tuple[int, ...] | tuple[Iterable[int], ...]
    )


class InitialMapping(Environment[Dict[str, NDArray[np.int_]], NDArray[np.int_]]):
    """RL environment for the initial mapping problem of OpenQL."""

    __slots__ = (
        "_rewarder",
        "_state",
        "observation_space",
        "action_space",
        "metadata",
        "_visualiser",
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        graph_generator: GraphGenerator | None = None,
        *,
        connection_graph: nx.Graph | None = None,
        connection_graph_matrix: ArrayLike | None = None,
        connection_grid_size: Gridspecs | None = None,
        rewarder: Rewarder | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the action space, observation space, and initial states.
        Furthermore, the connection graph and edge probability for the random
        interaction graph of each episode is defined.

        The supported render modes of this environment are ``"human"`` and
        ``"rgb_array"``.

        Args:
            interaction_graph_edge_probability: Probability that an edge between any
                pair of qubits in the random interaction graph exists. The interaction
                graph will have the same amount of nodes as the connection graph. Nodes
                without any interactions can be seen as 'null' nodes. Must be a value in
                the range [0,1].
            connection_graph: ``networkx`` graph representation of the QPU topology.
                Each node represents a physical qubit and each node represents a
                connection in the QPU topology.
            connection_graph_matrix: Adjacency matrix representation of the QPU
                topology.
            connection_grid_size: Size of the connection graph when the connection graph
                has a grid topology. For more information on the allowed values and
                types, see ``networkx`` `grid_graph`_ documentation.
            rewarder: Rewarder to use for the environment. Must inherit from
                :class:`qgym.templates.Rewarder`. If ``None`` (default), then
                :class:`qgym.envs.initial_mapping.BasicRewarder` is used.
            render_mode: If ``"human"`` open a ``pygame`` screen visualizing the step.
                If ``"rgb_array"``, return an RGB array encoding of the rendered frame
                on each render call.

        .. _grid_graph: https://networkx.org/documentation/stable/reference/generated/
            networkx.generators.lattice.grid_graph.html#grid-graph
        """
        # Check user input and parse it to a uniform format
        connection_graph = parse_connection_graph(
            connection_graph, connection_graph_matrix, connection_grid_size
        )

        if graph_generator is None:
            graph_generator = BasicGraphGenerator(len(connection_graph), seed=self.rng)
        else:
            check_instance(graph_generator, "graph_generator", GraphGenerator)
            if not graph_generator.finite:
                raise ValueError("'graph_generator' should not be a finite iterator")
            graph_generator = deepcopy(graph_generator)

        self._rewarder = parse_rewarder(rewarder, BasicRewarder)

        # Define internal attributes
        self._state = InitialMappingState(connection_graph, graph_generator)
        self.observation_space = self._state.create_observation_space()
        # Define attributes defined in parent class
        self.action_space = qgym.spaces.MultiDiscrete(
            nvec=[self._state.n_nodes, self._state.n_nodes], rng=self.rng
        )

        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self._visualiser = parse_visualiser(
            render_mode, InitialMappingVisualiser, [connection_graph]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.int_]], dict[str, Any]]:
        r"""Reset the state and set a new interaction graph.

        To be used after an episode is finished.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call i.e., before any learning is done.
            return_info: Whether to receive debugging info. Default is ``False``.
            options: Mapping with keyword arguments with additional options for the
                reset. Keywords can be found in the description of 
                :class:`~qgym.envs.initial_mapping.InitialMappingState`.\
                :func:`~qgym.envs.initial_mapping.InitialMappingState.reset()`.

        Returns:
            Initial observation and debugging info.
        """
        # call super method for dealing with the general stuff
        return super().reset(seed=seed, options=options)
