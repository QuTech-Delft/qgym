import networkx as nx
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class MappingEnvironment:
    """
    RL Mapping environment containing the current state of the problem.
    """
    
    def __init__(self, gridsize=(3,3), nmbr_log_qbits = 6, edge_p=0.5) -> None:
        """
        Initialize the action space, state space and initial state

        :param initialisation_input: this could for example be cQASM file
        """
                
        # Create the topology graph
        G_T = nx.generators.lattice.grid_graph(gridsize)
        self.A   = nx.linalg.graphmatrix.adjacency_matrix(G_T)
        
        # Create a random connection graph with nmbr_log_qbits nodes and 
        # edge probability edge_p
        G_C = nx.fast_gnp_random_graph(nmbr_log_qbits,edge_p)
        self.B   = nx.linalg.graphmatrix.adjacency_matrix(G_C)
        
        NUM_PHYSICAL_QUBITS = self.A.shape[0] 
        NUM_LOGICAL_QUBITS  = self.B.shape[0]
        
        
        # main attributes
        self._action_space = spaces.MultiDiscrete([NUM_PHYSICAL_QUBITS, NUM_LOGICAL_QUBITS])
        self._state_space  = spaces.Discrete(NUM_PHYSICAL_QUBITS)   
        self._state        = [-1] * NUM_PHYSICAL_QUBITS

        # other attributes
        self.step_number = 0
        # todo: more?

    @property
    def state(self):
        """
        Current state of the environment.
        """
        return self._state

    @property
    def state_space(self):
        """
        Description of the full state space, i.e. collection of all possible states.
        """
        return self._state_space

    @property
    def action_space(self):
        """
        Description of the current action space, i.e. the set of allowed actions (potentially depending on the current
        state.
        """
        return self._action_space

    def step(self, action):
        """
        Compute a new state based on the input action. Also give a reward, done-indicator and (optional) debugging info
        based on the updated state.

        :param action: Valid action to take.
        :return: A tuple containing four entries: 1) The updated state; 2) Reward of the new state; 3) Boolean value
            stating whether the new state is a final state (i.e. if we are done); 4) Additional (debugging) information.
        """
        # Increase the step number
        self.step_number += 1

        # Update the current state
        self._update_state(action)

        # Compute the reward, done indicator and define debugging info
        reward = self._compute_reward()
        done = self._is_done()
        info = {"step_nmbr": self.step_number}
        return self._state, reward, done, info

    def random_action(self):
        """
        Generate a random action from the action space

        :return: Random action.
        """
        # Determine which physical and logical qubits are not yet mapped
        available_physical_qubits, available_logical_qubits = self.available_actions()
        
        # Choose random indexes
        rnd_physical_idx = np.random.randint(len(available_physical_qubits))
        rnd_logical_idx  = np.random.randint(len(available_logical_qubits))
        
        # Generate the random action from the action space
        action = (available_physical_qubits[rnd_physical_idx], available_logical_qubits[rnd_logical_idx])
        return action

    def reset(self, optional_arguments = None):
        """
        Reset the environment to the initial state. To be used after an episode
        is finished. Optionally, one can implement increasingly more difficult
        environments for curriculum learning, or give new problem altogether.

        :param optional_arguments: turn on curriculum learning or give new problem
        :return: State after reset.
        """
        # Reset to the initial state and step number
        self._state = [-1] * self.state_space.n
        self.step_number = 0


        return self._state

    def _update_state(self, action) -> None:
        physical_qubits_idx = action[0]
        logical_qubits_idx  = action[1]
        self._state[physical_qubits_idx] = logical_qubits_idx # update state based on the given action


    def _compute_reward(self) -> float:
        """
        Give a reward based on the current state
        """
        mapping = self._state
        A = self.A
        B = self.B
        
        reward = 0.0  # compute a reward based on self.state
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if A[i,j] == 0 and B[mapping[i], mapping[j]] == 1:
                    reward -= 1
                if A[i,j] == 1 and B[mapping[i], mapping[j]] == 1:
                    reward += 1

        return reward

    def _is_done(self) -> bool:
        """
        Give a done-indicator based on the current state
        """
        done = (self.step_number >= self.action_space[1].n)  # determine if 'done' based on self.state
        return done
    
    def available_actions(self):
        available_physical_qubits = [i for i, c in enumerate(self._state) if c == -1]
        
        used_logical_qubits       = [c for i, c in enumerate(self._state) if c != -1]
        available_logical_qubits  = [i for i in range(self._action_space[1].n) if i not in used_logical_qubits]
        
        return available_physical_qubits, available_logical_qubits





def plot_mapping(MappingEnv):    
    
    # Check if the mapping is done
    if not MappingEnv._is_done():
        raise ValueError("MappingEnv is_done is not 'True'")
    
    A = MappingEnv.A
    B = MappingEnv.B
    mapping = MappingEnv.state
    
    # Generate the 'mapped' graph
    # gray edges are edges that are not used
    # green edges are present in both the connectivity graph as well as the embedded graph
    # red edges are present in the connectivity grapg, but not in the embedded graph
    G = nx.Graph()
    for i in range(A.shape[0]):
        G.add_node(i)
        
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if A[i,j] == 0 and B[mapping[i], mapping[j]] == 1:
                G.add_edge(i, j, color='red')
            if A[i,j] == 1 and B[mapping[i], mapping[j]] == 1:
                G.add_edge(i, j, color='green')
            if A[i,j] == 1 and B[mapping[i], mapping[j]] == 0:
                G.add_edge(i, j, color='gray')
    
    
    # Draw the 3 graphs
    fig, axes = plt.subplots(1,3, figsize=(12,6))
    nx.draw(nx.from_numpy_matrix(env.A), ax = axes[0])
    nx.draw(nx.from_numpy_matrix(env.B), ax = axes[1])
    
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    nx.draw(G, ax = axes[2], edge_color=colors)
    
    axes[0].set_title('Topology Graph')
    axes[1].set_title('Connectivity Graph')
    axes[2].set_title("'Mapped' Graph")
    
    plt.show()
    


env = MappingEnvironment()

for __ in range(3):
    observation = env.reset()
    for _ in range(100):
        action = env.random_action()
        state, reward, done, info = env.step(action)
        
        if done:
            plot_mapping(env)
            break


