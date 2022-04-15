"""
Environment and rewarder for training an RL agent on the initial mapping problem of OpenQL.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pygame
from networkx import Graph, fast_gnp_random_graph, grid_graph, to_scipy_sparse_matrix
from numpy.typing import NDArray
from pygame import gfxdraw

import qgym.spaces
from qgym.environment import Environment
from qgym.utils import check_adjacency_matrix
from qgym.envs.initial_mapping_rewarders import BasicRewarder

# Define some colors used during rendering
WHITE = (225, 225, 225)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
RED = (225, 0, 0)
GREEN = (0, 225, 0)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 0, 225)



class InitialMapping(
    Environment[Tuple[NDArray[np.int_], NDArray[np.int_]], NDArray[np.int_]]
):
    """
    RL environment for the initial mapping problem.
    """

    def __init__(
        self,
        connection_graph: Optional[Graph] = None,
        interaction_graph: Optional[Graph] = None,
        connection_graph_matrix: Optional[NDArray[Any]] = None,
        interaction_graph_matrix: Optional[NDArray[Any]] = None,
        connection_grid_size: Optional[Tuple[int, int]] = None,
        interaction_graph_edge_probability: Optional[float] = None,
    ) -> None:
        """
        Initialize the action space, observation space, and initial states. This also defines the connection and
        random interaction graph based on the arguments.

        :param connection_graph: networkx graph representation of the QPU topology
        :param interaction_graph: networkx graph representation of the interactions in a quantum circuit
        :param connection_graph_matrix: adjacency matrix representation of the QPU topology
        :param interaction_graph_matrix: adjacency matrix representation of the interactions in a quantum circuit
        :param connection_grid_size: Size of the connection graph. We only support grid-shaped connection graphs at the
            moment.
        :param interaction_graph_edge_probability: Probability that an edge between any pair of qubits in the random
            interaction graph exists. the interaction graph will have the same number of nodes as the connection graph.
            Nodes without any interactions can be seen as 'null' nodes.
        """
        if connection_graph is not None and interaction_graph is not None:
            (
                self._connection_graph,
                self._interaction_graph,
            ) = self._parse_network_graphs(connection_graph, interaction_graph)
        elif (
            connection_graph_matrix is not None and interaction_graph_matrix is not None
        ):
            (
                self._connection_graph,
                self._interaction_graph,
            ) = self._parse_adjacency_matrices(
                connection_graph_matrix, interaction_graph_matrix
            )
        elif (
            connection_grid_size is not None
            and interaction_graph_edge_probability is not None
        ):
            # Generate connection grid graph
            self._connection_graph: Graph = grid_graph(connection_grid_size)
            self._interaction_graph_edge_probability = (
                interaction_graph_edge_probability
            )

            # Create a random connection graph with `num_nodes` and with edges existing with probability
            # `interaction_graph_edge_probability` (nodes without connections can be seen as 'null' nodes)
            self._interaction_graph = fast_gnp_random_graph(
                self._connection_graph.number_of_nodes(),
                interaction_graph_edge_probability,
            )
        else:
            raise ValueError(
                "No valid arguments for instantiation of this environment were provided."
            )

        # Define internal attributes
        self._state = {
            "connection_graph_matrix": to_scipy_sparse_matrix(self._connection_graph),
            "num_nodes": self._connection_graph.number_of_nodes(),
            "interaction_graph_matrix": to_scipy_sparse_matrix(
                self._interaction_graph
            ).toarray(),
            "steps_done": 0,
            "mapping": np.full(self._connection_graph.number_of_nodes(), self._connection_graph.number_of_nodes()),
            "mapping_dict": {},
            "physical_qubits_mapped": set(),
            "logical_qubits_mapped": set(),
        }

        # Define attributes defined in parent class
        mapping_space = qgym.spaces.MultiDiscrete(
            nvec=[
                self._state["num_nodes"] + 1 for _ in range(self._state["num_nodes"])
            ],
            rng=self.rng,
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
            nvec=[
                self._state["num_nodes"],
                self._state["num_nodes"],
            ],
            rng=self.rng,
        )
        self._rewarder = BasicRewarder()
        self.metadata = {"render.modes": ["human"]}

        # Rendering data
        self.screen = None
        self.is_open = False
        self.screen_width = 1300
        self.screen_height = 730
        self.padding = 10

    def reset(
        self, *, seed: Optional[int] = None, return_info: bool = False, **_kwargs: Any
    ) -> Union[
        Tuple[NDArray[np.int_], NDArray[np.int_]],
        Tuple[Tuple[NDArray[np.int_], NDArray[np.int_]], Dict[Any, Any]],
    ]:
        """
        Reset state, action space and step number and load a new random initial state. To be used after an episode
        is finished.

        :param seed: Seed for the random number generator, should only be provided (optionally) on the first reset call,
            i.e. before any learning is done.
        :param return_info: Whether to receive debugging info.
        :param _kwargs: Additional options to configure the reset.
        :return: Initial observation and optional debugging info.
        """

        # Reset the state, action space, and step number
        self._interaction_graph = fast_gnp_random_graph(
            self._connection_graph.number_of_nodes(),
            self._interaction_graph_edge_probability,
        )
        self._state["interaction_graph_matrix"] = to_scipy_sparse_matrix(
            self._interaction_graph
        ).toarray()
        self._state["steps_done"] = 0
        self._state["mapping"] = np.full(self._state["num_nodes"], self._state["num_nodes"])
        self._state["mapping_dict"] = {}
        self._state["physical_qubits_mapped"] = set()
        self._state["logical_qubits_mapped"] = set()

        # call super method for dealing with the general stuff
        return super().reset(seed=seed, return_info=return_info)

    def render(self, mode: str = "human") -> bool:
        """
        Render the current state using pygame. The upper left screen shows the
        connection graph. The lower left screen the interaction graph. The
        right screen shows the mapped graph. Gray edges are unused, green edges
        are mapped correctly and red edges need at least on swap.

        :param mode: The mode to render with (default is 'human')
        """
        if mode not in self.metadata["render.modes"]:
            raise ValueError("The given render mode is not supported.")

        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Mapping Environment")
            self.is_open = True

        pygame.time.delay(10)

        self.screen.fill(GRAY)

        # draw screens for testing
        small_screen_dim = (
            self.screen_width / 2 - 1.5 * self.padding,
            self.screen_height / 2 - 1.5 * self.padding,
        )
        large_screen_dim = (
            self.screen_width / 2 - 1.5 * self.padding,
            self.screen_height - 2 * self.padding,
        )

        screen1_pos = (self.padding, self.padding)
        screen2_pos = (self.padding, small_screen_dim[1] + 2 * self.padding)
        screen3_pos = (small_screen_dim[0] + 2 * self.padding, self.padding)

        subscreen1 = pygame.draw.rect(
            self.screen,
            WHITE,
            [screen1_pos[0], screen1_pos[1], small_screen_dim[0], small_screen_dim[1]],
        )
        subscreen2 = pygame.draw.rect(
            self.screen,
            WHITE,
            [screen2_pos[0], screen2_pos[1], small_screen_dim[0], small_screen_dim[1]],
        )
        subscreen3 = pygame.draw.rect(
            self.screen,
            WHITE,
            [screen3_pos[0], screen3_pos[1], large_screen_dim[0], large_screen_dim[1]],
        )

        mapped_graph = self._get_mapped_graph()

        self._draw_graph(self._connection_graph, subscreen1)
        self._draw_graph(self._interaction_graph, subscreen2)
        self._draw_graph(mapped_graph, subscreen3, pivot_graph=self._connection_graph)

        pygame.event.pump()
        pygame.display.flip()

        return self.is_open

    def add_random_edge_weights(self) -> None:
        """
        Add random weights to the connection graph and interaction graph
        """

        for (u, v) in self._connection_graph.edges():
            self._connection_graph.edges[u, v]["weight"] = self.rng.gamma(2, 2) / 4
        self._state["connection_graph_matrix"] = to_scipy_sparse_matrix(
            self._connection_graph
        )

        for (u, v) in self._interaction_graph.edges():
            self._interaction_graph.edges[u, v]["weight"] = self.rng.gamma(2, 2) / 4
        self._state["interaction_graph_matrix"] = to_scipy_sparse_matrix(
            self._interaction_graph
        )

    def close(self):
        """
        Closed the screen used for rendering
        """
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            self.screen = None

    def _update_state(self, action: NDArray[np.int_]) -> None:
        """
        Update the state of this environment using the given action.

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

    def _compute_reward(
        self,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        *_args: Any,
        **_kwargs: Any,
    ) -> float:
        """
        Asks the rewarder to compute a reward, given the current state.
        """
        return super()._compute_reward(
            old_state=old_state, action=action, new_state=self._state
        )

    def _obtain_observation(self) -> Dict[str, NDArray[np.int_]]:
        """
        :return: Observation based on the current state.
        """
        return {
            "mapping": self._state["mapping"],
            "interaction_matrix": self._state["interaction_graph_matrix"].flatten(),
        }

    def _is_done(self) -> bool:
        """
        :return: Boolean value stating whether we are in a final state.
        """
        return len(self._state["physical_qubits_mapped"]) == self._state["num_nodes"]

    def _obtain_info(self) -> Dict[Any, Any]:
        """
        :return: Optional debugging info for the current state.
        """
        return {"Steps done": self._state["steps_done"]}

    @staticmethod
    def _parse_network_graphs(
        connection_graph: Graph, interaction_graph: Graph
    ) -> Tuple[Graph, Graph]:
        """
        Parse a given interaction and connection graph to the correct format.

        :param connection_graph: networkx graph representation of the QPU topology
        :param interaction_graph: networkx graph representation of the interactions in a quantum circuit
        """

        if not (
            isinstance(connection_graph, Graph) and isinstance(interaction_graph, Graph)
        ):
            raise TypeError(
                "The connection graph and interaction graph must both be of type Graph."
            )

        # deepcopy the graphs for safety
        connection_graph = deepcopy(connection_graph)
        interaction_graph = deepcopy(interaction_graph)

        n_connection = connection_graph.number_of_nodes()
        if interaction_graph.number_of_nodes() > n_connection:
            raise ValueError(
                f"The number of nodes in the interaction graph ({interaction_graph.number_of_nodes()}) "
                f"should be smaller than or equal to the number of nodes in the connection graph "
                f"({n_connection})"
            )

        # extend the interaction graph to the proper size
        null_index = 0
        while interaction_graph.number_of_nodes() < n_connection:
            interaction_graph.add_node(f"null_{null_index}")
            null_index += 1

        return connection_graph, interaction_graph

    @staticmethod
    def _parse_adjacency_matrices(
        connection_graph_matrix: NDArray[Any], interaction_graph_matrix: NDArray[Any]
    ) -> Tuple[Graph, Graph]:
        """
        Parse a given interaction and connection adjacency matrix to their respective graphs.

        :param connection_graph_matrix: adjacency matrix representation of the QPU topology
        :param interaction_graph_matrix: adjacency matrix representation of the interactions in a quantum circuit
        """
        if not check_adjacency_matrix(
            connection_graph_matrix
        ) or not check_adjacency_matrix(interaction_graph_matrix):
            raise TypeError(
                "Both the connection and interaction graph adjacency matrices should be square 2-D Numpy arrays."
            )

        # Construct an extended interaction matrix
        n_interaction = interaction_graph_matrix.size[0]
        extended_interaction_graph_matrix = np.zeros_like(connection_graph_matrix)
        extended_interaction_graph_matrix[
            :n_interaction, :n_interaction
        ] = interaction_graph_matrix

        connection_graph = nx.from_numpy_array(connection_graph_matrix)
        interaction_graph = nx.from_numpy_array(extended_interaction_graph_matrix)
        return connection_graph, interaction_graph

    def _get_mapped_graph(self) -> Graph:
        """
        Constructs a mapped graph. In this graph gray edges are unused, green
        edges are mapped correctly and red edges need at least on swap. This
        function is used during rendering.

        :return: Mapped graph
        """
        mapping = self._state["mapping_dict"]

        # Make the adjacency matrix of the mapped graph
        mapped_adjacency_matrix = np.zeros(self._state["connection_graph_matrix"].shape)
        for map_i, i in mapping.items():
            for map_j, j in mapping.items():
                mapped_adjacency_matrix[i, j] = self._state["interaction_graph_matrix"][
                    map_i, map_j
                ]

        # Make a networkx graph of the mapped graph
        graph = nx.Graph()
        for i in range(self._state["connection_graph_matrix"].shape[0]):
            graph.add_node(i)

        for i in range(self._state["connection_graph_matrix"].shape[0]):
            for j in range(self._state["connection_graph_matrix"].shape[1]):
                self._add_colored_edge(graph, mapped_adjacency_matrix, (i, j))

        # Relabel nodes for drawing
        nodes_mapping = dict(
            [(i, node) for i, node in enumerate(self._connection_graph.nodes)]
        )
        graph = nx.relabel_nodes(graph, nodes_mapping)

        return graph


    def _add_colored_edge(
        self, graph: Graph, mapped_adjacency_matrix: NDArray, edge: Tuple[int, int]
    ) -> None:
        """
        Utility function for making the mapped graph. Gives and edge of the
        graph a certain color. Gray edges are unused, green edges are mapped
        correctly and red edges need at least on swap.

        :param graph: The graph of which the edges must be colored
        :param mapped_adjacency_matrix: the adjacency matrix of the mapped graph
        :param edge: The edge that will be colored
        """
        (i, j) = edge
        is_connected = self._state["connection_graph_matrix"][i, j] != 0
        is_mapped = mapped_adjacency_matrix[i, j] != 0
        if not is_connected and is_mapped:
            graph.add_edge(i, j, color="red")
        if is_connected and is_mapped:
            graph.add_edge(i, j, color="green")
        if is_connected and not is_mapped:
            graph.add_edge(i, j, color="gray")

    def _draw_graph(
        self, graph: Graph, subscreen: pygame.Rect, pivot_graph: Optional[Graph] = None
    ) -> None:
        """
        Draws a graph on one of the subscreens.

        :param graph: the graph to be drawn
        :param subscreen: the subscreen on which the graph must be drawn
        :param pivot_graph: optional graph for which the spectral structure
            will be used for visualisation
        """
        if pivot_graph is None:
            node_positions = self._get_render_positions(graph, subscreen)
        else:
            assert graph.nodes == pivot_graph.nodes
            node_positions = self._get_render_positions(pivot_graph, subscreen)

        # Draw the nodes of the graph on the subscreen
        for node, pos in node_positions.items():
            gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 5, BLUE)
        # Draw the edges of the graph on the subscreen
        for (u, v) in graph.edges():
            pos_u = node_positions[u]
            pos_v = node_positions[v]
            color = BLACK
            if "color" in graph.edges[u, v]:
                if graph.edges[u, v]["color"] == "red":
                    color = RED
                if graph.edges[u, v]["color"] == "green":
                    color = GREEN
                if graph.edges[u, v]["color"] == "gray":
                    color = DARK_GRAY
            pygame.draw.aaline(self.screen, color, pos_u, pos_v)

    @staticmethod
    def _get_render_positions(
        graph: Graph, subscreen: pygame.Rect
    ) -> Dict[Any, Tuple[float, float]]:
        """
        Utility function used during render. Give the positions of the nodes
        of a graph on a given subscreen.

        :param graph: the graph of which the node positions must be determined
        :param subscreen: the subscreen on which the graph will be drawn
        :return: a dictionary where the keys are the names of the nodes and the
            values are the coordinates of these nodes
        """
        x_scaling = 0.45 * subscreen.width
        y_scaling = 0.45 * subscreen.height
        x_offset = subscreen.centerx
        y_offset = subscreen.centery
        node_positions = nx.spectral_layout(graph)
        for node in node_positions:
            node_positions[node][0] += node_positions[node][0] * x_scaling + x_offset
            node_positions[node][1] += node_positions[node][1] * y_scaling + y_offset
        return node_positions
