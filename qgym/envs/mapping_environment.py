from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import (
    Graph,
    adjacency_matrix,
    fast_gnp_random_graph,
    from_numpy_matrix,
    grid_graph,
)
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from qgym import Rewarder
from qgym.environment import Environment
from qgym.spaces import AdaptiveMultiDiscrete, InjectivePartialMap


class BasicRewarder(Rewarder):
    """
    Basic rewarder for the InitialMapping environment.
    """

    def __init__(self):
        """
        Initializes a new BasicRewarder.
        """
        self._reward_range = (-float("inf"), float("inf"))

    def compute_reward(
        self,
        state: NDArray[np.int_],
        connection_graph_matrix: csr_matrix,
        interaction_graph_matrix: csr_matrix,
    ):
        """
        Compute a reward, based on the current state, and the connection and interaction graphs.

        :param state: Current state of the InitialMapping
        :param connection_graph_matrix: Adjacency matrix of the connection graph of the InitialMapping Environment.
        :param interaction_graph_matrix: Adjacency matrix of the interaction graph of the InitialMapping Environment.
        """
        reward = 0.0  # compute a reward based on self.state
        for i in range(connection_graph_matrix.shape[0]):
            for j in range(connection_graph_matrix.shape[0]):
                if (
                    connection_graph_matrix[i, j] == 0
                    and interaction_graph_matrix[state[i], state[j]] != 0
                ):
                    reward -= 1
                if (
                    connection_graph_matrix[i, j] != 0
                    and interaction_graph_matrix[state[i], state[j]] != 0
                ):
                    reward += 1
        return reward


class InitialMapping(Environment[NDArray[np.int_], NDArray[np.int_]]):
    """
    RL environment for the initial mapping problem.
    """

    def __init__(
        self,
        connection_grid_size: Tuple[int, int] = (3, 3),
        interaction_graph_qubits: int = 6,
        interaction_graph_edge_probability: float = 0.5,
    ) -> None:
        """
        Initialize the action space, observation space, and initial states. This also defines the connection and
        random interaction graph based on the arguments.

        :param connection_grid_size: Size of the connection graph. We only support grid-shaped connection graphs at the
            moment.
        :param interaction_graph_qubits: Number of qubits of the random interaction graph.
        :param interaction_graph_edge_probability: Probability that an edge between any pair of qubits in the random
            interaction graph exists.
        """

        # Interaction graph
        self._connection_graph: Graph = grid_graph(connection_grid_size)
        self._connection_graph_matrix = adjacency_matrix(self._connection_graph)

        # Create a random connection graph with `interaction_graph_qubits` nodes and with edges existing with
        # probability `interaction_graph_edge_probability`
        self._interaction_graph = fast_gnp_random_graph(
            interaction_graph_qubits, interaction_graph_edge_probability
        )
        self._interaction_graph_matrix = adjacency_matrix(self._interaction_graph)

        # Define attributes defined in parent class
        self._observation_space = InjectivePartialMap(
            domain_size=self._interaction_graph.number_of_nodes(),
            codomain_size=self._connection_graph.number_of_nodes(),
        )
        self._action_space: AdaptiveMultiDiscrete = AdaptiveMultiDiscrete(
            sizes=[
                self._connection_graph.number_of_nodes(),
                self._interaction_graph.number_of_nodes(),
            ],
            starts=[0, 0],
            rng=self.rng,
        )
        self._rewarder = BasicRewarder()
        # todo: metadata (probably for rendering options)

        # Define internal attributes
        self._state = np.full(self._connection_graph.number_of_nodes(), -1)
        self._steps_done = 0
        self._max_steps = self._interaction_graph.number_of_nodes()

    @classmethod
    def from_networkx_graph(cls, connection_graph: Graph, interaction_graph: Graph):
        """
        Initialize the action space, observation space, and initial states
        according to given networkx graphs

        :param connection_graph: networkx graph representation of the QPU topology
        :param interaction_graph: networkx graph representation of the interactions in a quantum circuit
        """

        if not (
            isinstance(connection_graph, Graph) and isinstance(interaction_graph, Graph)
        ):
            raise TypeError(
                "connection_graph and interaction_graph must be of type Graph"
            )

        n_connection = connection_graph.number_of_nodes()
        n_interaction = interaction_graph.number_of_nodes()
        mapping_env = cls(
            connection_grid_size=(n_connection, 1),
            interaction_graph_qubits=n_interaction,
        )

        mapping_env._connection_graph = connection_graph
        mapping_env._connection_graph_matrix = adjacency_matrix(connection_graph)

        mapping_env._interaction_graph = interaction_graph
        mapping_env._interaction_graph_matrix = adjacency_matrix(interaction_graph)
        return mapping_env

    @classmethod
    def from_adjacency_matrix(
        cls, connection_graph_matrix: csr_matrix, interaction_graph_matrix: csr_matrix
    ):
        """
        Initialize the action space, observation space, and initial states
        according to given adjacency matrix representations of the graphs.

        :param connection_graph_matrix: adjacency matrix representation of the QPU topology
        :param interaction_graph_matrix: adjacency matrix representation of the interactions in a quantum circuit
        """
        try:
            connection_graph_matrix[0, 0]
            interaction_graph_matrix[0, 0]
        except:
            raise TypeError(
                "connection_graph_matrix and interaction_graph_matrix must be matrix like"
            )

        n_connection = connection_graph_matrix.shape[0]
        n_interaction = interaction_graph_matrix.shape[0]

        mapping_env = cls(
            connection_grid_size=(n_connection, 1),
            interaction_graph_qubits=n_interaction,
        )

        mapping_env._connection_graph_matrix = connection_graph_matrix
        mapping_env._connection_graph = from_numpy_matrix(connection_graph_matrix)

        mapping_env._interaction_graph_matrix = interaction_graph_matrix
        mapping_env._interaction_graph = from_numpy_matrix(interaction_graph_matrix)
        return mapping_env

    def step(
        self, action: NDArray[np.int_], *, return_info: bool = False
    ) -> Union[
        Tuple[NDArray[np.int_], float, bool],
        Tuple[NDArray[np.int_], float, bool, Dict[Any, Any]],
    ]:
        """
        Update the mapping based on the map action. Return observation, reward, done-indicator and (optional) debugging
        info based on the updated state.

        :param action: Valid action to take.
        :param return_info: Whether to receive debugging info.
        :return: A tuple containing three/four entries: 1) The updated state; 2) Reward of the new state; 3) Boolean
            value stating whether the new state is a final state (i.e. if we are done); 4) Optional Additional
            (debugging) information.
        """
        self._action_space.update(action)
        return super().step(action, return_info=return_info)

    def reset(  # todo we can use addiotional keyword-arguments to specify options for reset
        self, *, seed: Optional[int] = None, return_info: bool = False, **_kwargs: Any
    ) -> Union[NDArray[np.int_], Tuple[NDArray[np.int_], Dict[Any, Any]]]:
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
        self._state = np.full(self._connection_graph.number_of_nodes(), -1)
        self._action_space.reset()
        self._steps_done = 0

        # todo: new random graphs

        # call super method for dealing with the general stuff
        return super().reset(seed=seed, return_info=return_info)

    def render(self):
        """
        Render the current state.
        """
        raise NotImplementedError  # todo: implement this (probably based on plotting code below)

    def add_random_edge_weights(self) -> None:
        """
        Add random weights to the connection graph and interaction graph
        :param seed: Seed for the random number generator, should only be provided (optionally) on the first reset call,
            i.e. before any learning is done.
        """

        for (u, v) in self._connection_graph.edges():
            self._connection_graph.edges[u, v]["weight"] = self.rng.gamma(2, 2) / 4
        self._connection_graph_matrix = adjacency_matrix(self._connection_graph)

        for (u, v) in self._interaction_graph.edges():
            self._interaction_graph.edges[u, v]["weight"] = self.rng.gamma(2, 2) / 4
        self._interaction_graph_matrix = adjacency_matrix(self._interaction_graph)

    def _update_state(self, action: NDArray[np.int_]) -> None:
        """
        Update the state of this environment using the given action.

        :param action: Mapping action to be executed.
        """
        # Increase the step number
        self._steps_done += 1

        # update state based on the given action
        physical_qubit_index = action[0]
        logical_qubit_index = action[1]
        self._state[physical_qubit_index] = logical_qubit_index

    def _compute_reward(self) -> float:
        """
        Asks the rewarder to compute a reward, given the current state.
        """
        return super()._compute_reward(
            self._state, self._connection_graph_matrix, self._interaction_graph_matrix
        )

    def _obtain_observation(self) -> NDArray[np.int_]:
        """
        :return: Observation based on the current state.
        """
        return self._state

    def _is_done(self) -> bool:
        """
        :return: Boolean value stating whether we are in a final state.
        """
        return self._steps_done >= self._max_steps

    def _obtain_info(self) -> Dict[Any, Any]:
        """
        :return: Optional debugging info for the current state.
        """
        return {"Steps done": self._steps_done}


# todo: maybe move part of this to render and some clean-up
if __name__ == "__main__":

    def plot_mapping(MappingEnv):
        # Check if the mapping is done
        if not MappingEnv._is_done():
            raise ValueError("MappingEnv is_done is not 'True'")

        A = MappingEnv._connection_graph_matrix
        B = MappingEnv._interaction_graph_matrix
        mapping = MappingEnv._state

        # Generate the 'mapped' graph
        # gray edges are edges that are not used
        # green edges are present in both the connectivity graph as well as the embedded graph
        # red edges are present in the connectivity grapg, but not in the embedded graph
        G = nx.Graph()
        for i in range(A.shape[0]):
            G.add_node(i)

        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if A[i, j] == 0 and B[mapping[i], mapping[j]] == 1:
                    G.add_edge(i, j, color="red")
                if A[i, j] == 1 and B[mapping[i], mapping[j]] == 1:
                    G.add_edge(i, j, color="green")
                if A[i, j] == 1 and B[mapping[i], mapping[j]] == 0:
                    G.add_edge(i, j, color="gray")

        # Draw the 3 graphs
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        nx.draw(nx.from_numpy_matrix(env._connection_graph_matrix), ax=axes[0])
        nx.draw(nx.from_numpy_matrix(env._interaction_graph_matrix), ax=axes[1])

        edges = G.edges()
        colors = [G[u][v]["color"] for u, v in edges]
        nx.draw(G, ax=axes[2], edge_color=colors)

        axes[0].set_title("Topology Graph")
        axes[1].set_title("Connectivity Graph")
        axes[2].set_title("'Mapped' Graph")

        plt.show()

    env = InitialMapping()

    for __ in range(3):
        observation = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            state, reward, done = env.step(action)

            if done:
                plot_mapping(env)
                break
