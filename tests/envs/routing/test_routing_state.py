import networkx as nx
import numpy as np
import pytest

from qgym.envs.routing.routing_state import RoutingState


@pytest.fixture
def quad_graph():
    quad_graph = nx.Graph()
    quad_graph.add_edge(0, 1)
    quad_graph.add_edge(1, 2)
    quad_graph.add_edge(2, 3)
    quad_graph.add_edge(3, 0)
    return quad_graph

@pytest.fixture
def max_observation_reach(max_interaction_gates):
    return np.min(5, max_interaction_gates)

@pytest.mark.parametrize(
    "max_interaction_gates",
    [1, 5, 10, 50]
)

def test_routing_state_initialize(
    max_interaction_gates,
):
    state = RoutingState(
        max_interaction_gates= max_interaction_gates, 
        max_observation_reach= max_observation_reach(max_interaction_gates), 
        connection_graph= quad_graph(),
        observation_booleans_flag=False,
        observation_connection_flag=False,
        )
    assert True