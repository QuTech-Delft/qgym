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
    The state space is described by a dictionary with the following structure:

    * ``num_nodes``: Number of *physical* qubits.
    * ``connection_graph_matrix``: Sparse adjacency matrix of the connection graph.
    * ``interaction_graph_matrix``: Sparse adjacency matrix of the interaction graph.
      (Should have the same number of nodes as the connection graph.)
    * ``steps_done``: Number of steps done since the last reset.
    * ``mapping``: Array of which the index represents a physical qubit, and the value a
      virtual qubit (is set to ``num_nodes + 1`` when nothing is mapped to the physical
      qubit yet).
    * ``mapping_dict``: Dictionary that maps logical qubits to physical qubit.
    * ``physical_qubits_mapped``: Set containing all mapped physical qubits.
    * ``virtual_qubits_mapped``: Set containing all mapped virtual qubits.

Observation Space:
    The observation space is a ``qgym.spaces.Dict`` with 2 entries:

    * ``mapping``: The current state of the mapping.
    * ``interaction_matrix``: The flattened adjacency matrix of the interaction graph.

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

    By default,  ``InitialMapping`` uses the ``BasicRewarder``. As an example, we would
    like to change the rewarder to the ``EpisodeRewarder``. This can be done in the
    following way:

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
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import qgym.spaces
from networkx import Graph, fast_gnp_random_graph, grid_graph, to_scipy_sparse_array
from numpy.typing import ArrayLike, NDArray
from qgym.environment import Environment
from qgym.envs.initial_mapping.initial_mapping_rewarders import BasicRewarder
from qgym.envs.initial_mapping.initial_mapping_visualiser import (
    InitialMappingVisualiser,
)
from qgym.rewarder import Rewarder
from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_graph_is_valid_topology,
    check_instance,
    check_real,
    check_string,
)

Gridspecs = Union[List[Union[int, Iterable[int]]], Tuple[Union[int, Iterable[int]]]]


class InitialMapping(Environment[Dict[str, NDArray[np.int_]], NDArray[np.int_]]):
    """RL environment for the initial mapping problem of OpenQL."""

    def __init__(
        self,
        interaction_graph_edge_probability: float,
        connection_graph: Optional[Graph] = None,
        connection_graph_matrix: Optional[ArrayLike] = None,
        connection_grid_size: Optional[Gridspecs] = None,
        rewarder: Optional[Rewarder] = None,
    ) -> None:
        """Initialize the action space, observation space, and initial states.
        Furthermore, the connection graph and edge probability for the random
        interaction graph of each episode is defined.

        The supported render modes of this environment are "human" and "rgb_array".

        :param interaction_graph_edge_probability: Probability that an edge between any
            pair of qubits in the random interaction graph exists. The interaction
            graph will have the same amount of nodes as the connection graph. Nodes
            without any interactions can be seen as 'null' nodes. Must be a value in the
            range [0,1].
        :param connection_graph: ``networkx`` graph representation of the QPU topology.
            Each node represents a physical qubit and each node represents a connection
            in the QPU topology.
        :param connection_graph_matrix: Adjacency matrix representation of the QPU
            topology.
        :param connection_grid_size: Size of the connection graph when the connection
            graph has a grid topology. For more information on the allowed values and
            types, see ``networkx`` `grid_graph`_ documentation.
        :param rewarder: Rewarder to use for the environment. Must inherit from
            ``qgym.Rewarder``. If ``None`` (default), then ``BasicRewarder`` is used.

        .. _grid_graph: https://networkx.org/documentation/stable/reference/generated/
            networkx.generators.lattice.grid_graph.html#grid-graph
        """
        self._interaction_graph_edge_probability = check_real(
            interaction_graph_edge_probability,
            "interaction_graph_edge_probability",
            l_bound=0,
            u_bound=1,
        )
        self._connection_graph = self._parse_connection_graph(
            connection_graph=connection_graph,
            connection_graph_matrix=connection_graph_matrix,
            connection_grid_size=connection_grid_size,
        )

        self._rewarder = self._parse_rewarder(rewarder)

        # Create a random connection graph with `num_nodes` and with edges existing with
        # probability `interaction_graph_edge_probability` (nodes without connections
        # can be seen as 'null' nodes)
        self._interaction_graph = fast_gnp_random_graph(
            self._connection_graph.number_of_nodes(),
            self._interaction_graph_edge_probability,
        )

        # Define internal attributes
        self._state = {
            "connection_graph_matrix": to_scipy_sparse_array(self._connection_graph),
            "num_nodes": self._connection_graph.number_of_nodes(),
            "interaction_graph_matrix": to_scipy_sparse_array(
                self._interaction_graph
            ).toarray(),
            "steps_done": 0,
            "mapping": np.full(
                self._connection_graph.number_of_nodes(),
                self._connection_graph.number_of_nodes(),
            ),
            "mapping_dict": {},
            "physical_qubits_mapped": set(),
            "logical_qubits_mapped": set(),
        }

        # Define attributes defined in parent class
        mapping_space = qgym.spaces.MultiDiscrete(
            nvec=[self._state["num_nodes"] + 1] * self._state["num_nodes"], rng=self.rng
        )
        interaction_matrix_space = qgym.spaces.Box(
            low=0,
            high=np.iinfo(np.int64).max,
            shape=(self._state["num_nodes"] * self._state["num_nodes"],),
            dtype=np.int64,
        )
        self.observation_space = qgym.spaces.Dict(
            rng=self.rng,
            mapping=mapping_space,
            interaction_matrix=interaction_matrix_space,
        )
        self.action_space = qgym.spaces.MultiDiscrete(
            nvec=[self._state["num_nodes"], self._state["num_nodes"]], rng=self.rng
        )

        self.metadata = {"render.modes": ["human", "rgb_array"]}

        self._visualiser = InitialMappingVisualiser(self._connection_graph)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        interaction_graph: Optional[Graph] = None,
        **_kwargs: Any,
    ) -> Union[
        Dict[str, NDArray[np.int_]],
        Tuple[Dict[str, NDArray[np.int_]], Dict[str, Any]],
    ]:
        """Reset the state and set a new interaction graph. To be used after an episode
        is finished.

        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call i.e., before any learning is done.
        :param return_info: Whether to receive debugging info. Default is ``False``.
        :param interaction_graph: Interaction graph to be used for the next iteration,
            if ``None`` a random interaction graph will be created.
        :param _kwargs: Additional options to configure the reset.
        :return: Initial observation and optionally debugging info.
        """
        # Reset the state, action space, and step number
        if interaction_graph is None:
            self._interaction_graph = fast_gnp_random_graph(
                self._connection_graph.number_of_nodes(),
                self._interaction_graph_edge_probability,
            )
        else:
            self._interaction_graph = interaction_graph
        self._state["interaction_graph_matrix"] = to_scipy_sparse_array(
            self._interaction_graph
        ).toarray()
        self._state["steps_done"] = 0
        self._state["mapping"] = np.full(
            self._state["num_nodes"], self._state["num_nodes"]
        )
        self._state["mapping_dict"] = {}
        self._state["physical_qubits_mapped"] = set()
        self._state["logical_qubits_mapped"] = set()

        # call super method for dealing with the general stuff
        return super().reset(seed=seed, return_info=return_info)

    def render(self, mode: str = "human") -> Any:
        """Render the current state using ``pygame``. The upper left screen shows the
        connection graph. The lower left screen the interaction graph. The right screen
        shows the mapped graph. Gray edges are unused, green edges are mapped correctly
        and red edges do not match.

        :param mode: The mode to render with (should be one of the supported
            'render.modes' in ``self.metadata``).
        :raise ValueError: If the mode is not supported.
        :return: The result of rendering the current state. Return type depends on the
            render mode.
        """
        mode = check_string(mode, "mode", lower=True)
        if mode not in self.metadata["render.modes"]:
            raise ValueError("The given render mode is not supported")

        return self._visualiser.render(self._state, self._interaction_graph, mode)

    def close(self) -> None:
        """Close the screen used for rendering."""
        if hasattr(self, "_visualiser"):
            self._visualiser.close()

    def add_random_edge_weights(self) -> None:
        """Add random weights to the connection graph and interaction graph."""
        for (u, v) in self._connection_graph.edges():
            self._connection_graph.edges[u, v]["weight"] = self.rng.gamma(2, 2) / 4
        self._state["connection_graph_matrix"] = to_scipy_sparse_array(
            self._connection_graph
        )

        for (u, v) in self._interaction_graph.edges():
            self._interaction_graph.edges[u, v]["weight"] = self.rng.gamma(2, 2) / 4
        self._state["interaction_graph_matrix"] = to_scipy_sparse_array(
            self._interaction_graph
        )

    def _update_state(self, action: NDArray[np.int_]) -> None:
        """Update the state of this environment using the given action.

        :param action: Mapping action to be executed.
        """
        # Increase the step number
        self._state["steps_done"] += 1

        # update state based on the given action
        physical_qubit_index = action[0]
        logical_qubit_index = action[1]
        if (
            physical_qubit_index not in self._state["physical_qubits_mapped"]
            and logical_qubit_index not in self._state["logical_qubits_mapped"]
        ):
            self._state["mapping"][physical_qubit_index] = logical_qubit_index
            self._state["mapping_dict"][logical_qubit_index] = physical_qubit_index
            self._state["physical_qubits_mapped"].add(physical_qubit_index)
            self._state["logical_qubits_mapped"].add(logical_qubit_index)

    def _obtain_observation(self) -> Dict[str, NDArray[np.int_]]:
        """:return: Observation based on the current state."""
        return {
            "mapping": self._state["mapping"],
            "interaction_matrix": self._state["interaction_graph_matrix"].flatten(),
        }

    def _is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        return len(self._state["physical_qubits_mapped"]) == self._state["num_nodes"]  # type: ignore[no-any-return] # this always gives a bool

    def _obtain_info(self) -> Dict[str, Any]:
        """:return: Optional debugging info for the current state."""
        return {"Steps done": self._state["steps_done"]}

    @staticmethod
    def _parse_connection_graph(
        *,
        connection_graph: Any,
        connection_graph_matrix: Any,
        connection_grid_size: Any,
    ) -> Graph:
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
                msg = "Both 'connection_graph' and 'connection_graph_matrix' were "
                msg += "given. Using 'connection_graph'."
                warnings.warn(msg)
            if connection_grid_size is not None:
                msg = "Both 'connection_graph' and 'connection_grid_size' were given. "
                msg += "Using 'connection_graph'."
                warnings.warn(msg)

            check_graph_is_valid_topology(connection_graph, "connection_graph")

            # deepcopy the graphs for safety
            return deepcopy(connection_graph)

        elif connection_graph_matrix is not None:
            if connection_grid_size is not None:
                msg = "Both 'connection_graph_matrix' and 'connection_grid_size' were "
                msg += "given. Using 'connection_graph_matrix'."
                warnings.warn(msg)
            return InitialMapping._parse_adjacency_matrix(connection_graph_matrix)
        elif connection_grid_size is not None:
            # Generate connection grid graph
            return grid_graph(connection_grid_size)

        msg = "No valid arguments for instantiation of the initial mapping environment "
        msg += "were provided."
        raise ValueError(msg)

    @staticmethod
    def _parse_rewarder(rewarder: Union[Rewarder, None]) -> Rewarder:
        """Parse the `rewarder` given by the user.

        :param rewarder: ``Rewarder`` to use for the environment. If ``None``, then the
            ``BasicRewarder`` with default settings is used.
        :return: Rewarder.
        """
        if rewarder is None:
            return BasicRewarder()
        check_instance(rewarder, "rewarder", Rewarder)
        return deepcopy(rewarder)

    @staticmethod
    def _parse_adjacency_matrix(connection_graph_matrix: ArrayLike) -> Graph:
        """Parse a given connection graph adjacency matrix to its respective graph.

        :param connection_graph_matrix: adjacency matrix representation of the QPU
            topology.
        :raise TypeError: When the provided matrix is not a valid adjacency matrix.
        :return: Graph representation of the adjacency matrix.
        """
        connection_graph_matrix = check_adjacency_matrix(connection_graph_matrix)

        return nx.from_numpy_array(connection_graph_matrix)
