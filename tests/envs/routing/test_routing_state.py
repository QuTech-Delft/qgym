import networkx as nx
import numpy as np
import pytest

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
    [(1, 1), (5, 5), (10, 5), (50, 5)]
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
    assert True