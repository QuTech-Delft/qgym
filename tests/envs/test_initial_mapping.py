from stable_baselines3.common.env_checker import check_env

from qgym.envs import InitialMapping
from qgym.envs.initial_mapping import BasicRewarder
import numpy as np
from scipy.sparse import csr_matrix

def test_validity() -> None:
    env = InitialMapping(
        connection_grid_size=(3, 3), interaction_graph_edge_probability=0.5
    )
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True





"""
Tests for the basic rewarder
"""


def test_initialisation() -> None:
    rewarder = BasicRewarder()
    assert True

def test_illegal_action() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {0},
                 "logical_qubits_mapped"  : {1}}
    action = np.array([0,1])
    new_state={"physical_qubits_mapped" : {0},
               "logical_qubits_mapped"  : {1}}
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == -10

def test_reward_first_step_no_edges() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {},
                 "logical_qubits_mapped"  : {}}
    action = np.array([0,1])
    new_state={"connection_graph_matrix" : csr_matrix([[0,0],[0,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,0],[0,0]]),
               "mapping" : np.array([0,1]),
               "physical_qubits_mapped" : {1},
               "logical_qubits_mapped"  : {0}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == 0

def test_first_step_fully_connected() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {},
                 "logical_qubits_mapped"  : {}}
    action = np.array([0,1])
    new_state={"connection_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "mapping" : np.array([0,1]),
               "physical_qubits_mapped" : {1},
               "logical_qubits_mapped"  : {0}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == 0

def test_last_step_fully_connected() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {0},
                 "logical_qubits_mapped"  : {0}}
    action = np.array([1,1])
    new_state={"connection_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "mapping" : np.array([1,2]),
               "physical_qubits_mapped" : {0,1},
               "logical_qubits_mapped"  : {0,1}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == 5

def test_first_step_full_empty() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {},
                 "logical_qubits_mapped"  : {}}
    action = np.array([1,1])
    new_state={"connection_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,0],[0,0]]),
               "mapping" : np.array([0,2]),
               "physical_qubits_mapped" : {1},
               "logical_qubits_mapped"  : {1}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == 0

def test_last_step_full_empty() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {0},
                 "logical_qubits_mapped"  : {0}}
    action = np.array([1,1])
    new_state={"connection_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,0],[0,0]]),
               "mapping" : np.array([1,2]),
               "physical_qubits_mapped" : {0,1},
               "logical_qubits_mapped"  : {0,1}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == 0

def test_first_step_empty_full() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {},
                 "logical_qubits_mapped"  : {}}
    action = np.array([1,1])
    new_state={"connection_graph_matrix" : csr_matrix([[0,0],[0,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "mapping" : np.array([0,2]),
               "physical_qubits_mapped" : {1},
               "logical_qubits_mapped"  : {1}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == 0

def test_last_step_empty_full() -> None:
    rewarder = BasicRewarder()
    old_state = {"physical_qubits_mapped" : {1},
                 "logical_qubits_mapped"  : {1}}
    action = np.array([0,0])
    new_state={"connection_graph_matrix" : csr_matrix([[0,0],[0,0]]),
               "interaction_graph_matrix" : csr_matrix([[0,1],[1,0]]),
               "mapping" : np.array([1,2]),
               "physical_qubits_mapped" : {0,1},
               "logical_qubits_mapped"  : {0,1}
               }
    
    reward = rewarder.compute_reward(old_state=old_state,
                            action=action,
                            new_state=new_state)
    
    assert reward == -1




