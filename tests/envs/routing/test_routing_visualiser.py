import time

import networkx as nx
import pytest

from qgym.envs.routing import Routing


@pytest.mark.skip(reason="This needs to be manually inspected")
def test_routing_visualiser(mode="human"):
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    env = Routing(connection_graph=connection_graph)
    env.reset()

    env.render(mode)
    for _ in range(1000):
        time.sleep(0.1)
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render(mode)
        if done:
            break


if __name__ == "__main__":
    test_routing_visualiser("rgb_array")
    test_routing_visualiser("human")
