import networkx as nx
import pytest

from qgym.envs.initial_mapping import BasicRewarder, EpisodeRewarder
from qgym.utils.input_parsing import parse_connection_graph, parse_rewarder


class TestParseRewarder:
    def test_default(self):
        output = parse_rewarder(None, BasicRewarder)
        assert isinstance(output, BasicRewarder)
        assert output is not BasicRewarder()

    def test_default(self):
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

    def test_parse_connection_graph(self, expected_output: nx.Graph) -> None:
        graph = nx.Graph()
        graph.add_edge(0, 1)
        output_graph = parse_connection_graph(graph=graph)
        assert nx.is_isomorphic(output_graph, expected_output)
        # Test if it is a copy
        assert graph is not output_graph

    def test_parse_connection_graph(self, expected_output: nx.Graph) -> None:
        output_graph = parse_connection_graph(matrix=[[0, 1], [1, 0]])
        assert nx.is_isomorphic(output_graph, expected_output)

    def test_parse_connection_graph(self, expected_output: nx.Graph) -> None:
        output_graph = parse_connection_graph(grid_size=(1, 2))
        assert nx.is_isomorphic(output_graph, expected_output)


class TestParseConnectionGraphWarnings:
    @pytest.mark.parametrize(
        "matrix,grid_size", [("test", None), (None, "test"), ("test", "test")]
    )
    def test_nx_graph_and_other(self, matrix, grid_size):
        small_graph = nx.Graph()
        small_graph.add_edge(0, 1)

        with pytest.warns(UserWarning):
            parse_connection_graph(small_graph, matrix, grid_size)

    def test_array_like_and_gridspec(self):
        with pytest.warns(UserWarning):
            parse_connection_graph(matrix=[[0, 1], [1, 0]], grid_size="test")


def test_parse_connection_graph_exception():
    with pytest.raises(ValueError):
        parse_connection_graph()
