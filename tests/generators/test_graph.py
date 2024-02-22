"""This module contains tests for the graph generation module."""

from collections.abc import Iterator

import networkx as nx
import pytest

from qgym.generators.graph import (
    BasicGraphGenerator,
    GraphGenerator,
    NullGraphGenerator,
)


def test_graph_generator() -> None:
    with pytest.raises(TypeError):
        GraphGenerator()


class TestNullGraphGenerator:

    @pytest.fixture(name="generator")
    def null_graph_generator_fixture(self) -> None:
        return NullGraphGenerator()

    def test_infinite(self, generator: NullGraphGenerator) -> None:
        assert not generator.finite

    def test_inheritance(self, generator: NullGraphGenerator) -> None:
        assert isinstance(generator, GraphGenerator)
        assert isinstance(generator, Iterator)

    def test_next(self, generator: NullGraphGenerator) -> None:
        graph = next(generator)
        assert isinstance(graph, nx.Graph)
        assert len(graph) == 0

    def test_iter(self, generator: NullGraphGenerator) -> None:
        for i, graph in enumerate(generator):
            assert isinstance(graph, nx.Graph)
            assert len(graph) == 0

            if i > 100:
                break


class TestBasicGraphGenerator:

    @pytest.fixture(name="simple_generator")
    def null_graph_generator_fixture(self) -> None:
        return BasicGraphGenerator(5)

    def test_infinite(self, simple_generator: BasicGraphGenerator) -> None:
        assert not simple_generator.finite

    def test_inheritance(self, simple_generator: BasicGraphGenerator) -> None:
        assert isinstance(simple_generator, GraphGenerator)
        assert isinstance(simple_generator, Iterator)

    def test_next(self, simple_generator: BasicGraphGenerator) -> None:
        graph = next(simple_generator)
        assert isinstance(graph, nx.Graph)
        assert len(graph) == 5
        assert nx.number_of_selfloops(graph) == 0
        assert not graph.is_directed()
        assert not graph.is_multigraph()

    def test_iter(self, simple_generator: BasicGraphGenerator) -> None:
        for i, graph in enumerate(simple_generator):
            assert isinstance(graph, nx.Graph)
            assert len(graph) == 5
            assert nx.number_of_selfloops(graph) == 0
            assert not graph.is_directed()
            assert not graph.is_multigraph()

            if i > 100:
                break

    def test_full_edge_probability(self) -> None:
        generator = BasicGraphGenerator(5, 1)
        graph = next(generator)

        assert isinstance(graph, nx.Graph)
        assert len(graph) == 5
        assert nx.is_isomorphic(graph, nx.complete_graph(5))

    def test_seed(self) -> None:
        generator1 = BasicGraphGenerator(10, seed=1)
        generator2 = BasicGraphGenerator(10, seed=1)
        generator3 = BasicGraphGenerator(10, seed=3)

        for _ in range(10):
            graph1 = next(generator1)
            graph2 = next(generator2)
            graph3 = next(generator3)

            assert nx.is_isomorphic(graph1, graph2)
            assert not nx.is_isomorphic(graph1, graph3)
