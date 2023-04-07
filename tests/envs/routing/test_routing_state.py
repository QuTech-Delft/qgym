import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

import qgym.spaces

from qgym.envs.routing.routing_state import RoutingState

# Arrange
@pytest.fixture
def quad_graph():
    quad_graph = nx.Graph()
    quad_graph.add_edge(0, 1)
    quad_graph.add_edge(1, 2)
    quad_graph.add_edge(2, 3)
    quad_graph.add_edge(3, 0)
    return quad_graph


@pytest.mark.parametrize(
    "max_interaction_gates, max_observation_reach",
    [(1, 1), (5, 5), (10, 5), (50, 5), (5, 10)]
)

def test_routing_state_initialize(
    max_interaction_gates,
    max_observation_reach,
    quad_graph
):
    state = RoutingState(
        max_interaction_gates= max_interaction_gates, 
        max_observation_reach= max_observation_reach, 
        connection_graph= quad_graph,
        observation_booleans_flag=False,
        observation_connection_flag=False,
        )
    assert state.max_interaction_gates == max_interaction_gates
    assert state.swap_gates_inserted == []
    assert state.position == 0
    assert state.max_observation_reach <= int(
            min(max_observation_reach, len(state.interaction_circuit))
        )
    assert len(state.interaction_circuit)>=0

@pytest.fixture
def quad_graph():
    quad_graph = nx.Graph()
    quad_graph.add_edge(0, 1)
    quad_graph.add_edge(1, 2)
    quad_graph.add_edge(2, 3)
    quad_graph.add_edge(3, 0)
    return quad_graph


@pytest.mark.parametrize(
    "max_interaction_gates, max_observation_reach",
    [(1, 1), (5, 5), (10, 5), (50, 5), (5, 10)]
)
def test_routing_state_initialize(
    max_interaction_gates,
    max_observation_reach,
    quad_graph
):
    state = RoutingState(
        max_interaction_gates= max_interaction_gates, 
        max_observation_reach= max_observation_reach, 
        connection_graph= quad_graph,
        observation_booleans_flag=False,
        observation_connection_flag=False,
        )
    observation_space = state.create_observation_space()
    #TOD: revise below
    #assert isinstance(observation_space, qgym.spaces.Dict[str, list])
    