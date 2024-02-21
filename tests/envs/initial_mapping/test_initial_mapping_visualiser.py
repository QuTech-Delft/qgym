"""Test for the ``InitialMappingVisualiser``."""

import time

import networkx as nx
import pytest

from qgym.envs.initial_mapping import InitialMapping


@pytest.mark.skip(reason="This needs to be manually inspected")
def test_initial_mapping_visualiser() -> None:
    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    env = InitialMapping(connection_graph=connection_graph, render_mode="human")
    env.reset()

    for _ in range(1000):
        time.sleep(0.1)
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        if done:
            break


if __name__ == "__main__":
    test_initial_mapping_visualiser()
