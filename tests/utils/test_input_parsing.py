from __future__ import annotations

import networkx as nx
import pytest
from numpy.random import Generator, default_rng

from qgym.envs.initial_mapping import BasicRewarder, EpisodeRewarder
from qgym.utils.input_parsing import (
    has_fidelity,
    parse_connection_graph,
    parse_rewarder,
    parse_seed,
)


class TestParseRewarder:
    def test_default1(self) -> None:
        output = parse_rewarder(None, BasicRewarder)
        assert isinstance(output, BasicRewarder)
        assert output is not BasicRewarder()

    def test_default2(self) -> None:
        input_rewarder = EpisodeRewarder()
        output = parse_rewarder(input_rewarder, BasicRewarder)
        assert isinstance(output, EpisodeRewarder)
        # Test deepcopy
        assert output is not input_rewarder


class TestParseConnectionGraph:
    @pytest.fixture(scope="class", name="expected_output")
    def expected_output_fixture(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_edge(0, 1)
        return graph

    def test_parse_connection_graph1(self, expected_output: nx.Graph) -> None:
        graph = nx.Graph()
        graph.add_edge(0, 1)
        output_graph = parse_connection_graph(graph=graph)
        assert nx.is_isomorphic(output_graph, expected_output)
        # Test if it is a copy
        assert graph is not output_graph

    def test_parse_connection_graph2(self, expected_output: nx.Graph) -> None:
        output_graph = parse_connection_graph(matrix=[[0, 1], [1, 0]])
        assert nx.is_isomorphic(output_graph, expected_output)

    def test_parse_connection_graph3(self, expected_output: nx.Graph) -> None:
        output_graph = parse_connection_graph(grid_size=(1, 2))
        assert nx.is_isomorphic(output_graph, expected_output)


class TestParseConnectionGraphWarnings:
    @pytest.mark.parametrize(
        "matrix,grid_size", [("test", None), (None, "test"), ("test", "test")]
    )
    def test_nx_graph_and_other(
        self, matrix: str | None, grid_size: str | None
    ) -> None:
        small_graph = nx.Graph()
        small_graph.add_edge(0, 1)

        with pytest.warns(UserWarning):
            parse_connection_graph(small_graph, matrix, grid_size)  # type: ignore[arg-type]

    def test_array_like_and_gridspec(self) -> None:
        with pytest.warns(UserWarning):
            parse_connection_graph(matrix=[[0, 1], [1, 0]], grid_size="test")  # type: ignore[arg-type]


def test_parse_connection_graph_exception() -> None:
    with pytest.raises(ValueError):
        parse_connection_graph()


class TestHasFidelity:
    def test_positive(self) -> None:
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 0.9), (1, 2, 0.8)])
        assert has_fidelity(graph)

    def test_negative(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        assert not has_fidelity(graph)


class TestParseSeed:

    def test_none(self) -> None:
        rng = parse_seed(None)
        assert isinstance(rng, Generator)

    def test_supports_int(self) -> None:
        rng = parse_seed(1.0)
        assert isinstance(rng, Generator)

    def test_generator(self) -> None:
        rng = default_rng()
        assert rng is parse_seed(rng)

    def test_expected_int_behaviour(self) -> None:
        rng1 = parse_seed(1)
        rng2 = parse_seed(1)
        rng3 = parse_seed(2)

        for _ in range(100):
            r_float1 = rng1.random()
            r_float2 = rng2.random()
            r_float3 = rng3.random()

            assert r_float1 == r_float2
            assert r_float2 != r_float3

    def test_type_error(self) -> None:
        msg = "seed must be a Generator, SupportsInt or None, but was of type "
        msg += "<class 'str'>"
        with pytest.raises(TypeError, match=msg):
            parse_seed("test")
