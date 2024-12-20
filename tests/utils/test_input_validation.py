from numbers import Number
from typing import Any

import networkx as nx
import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix

from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_bool,
    check_graph_is_valid_topology,
    check_instance,
    check_int,
    check_real,
    check_string,
    warn_if_negative,
    warn_if_positive,
)


class TestCheckReal:
    @pytest.mark.parametrize(
        ("x_value", "expected_output"),
        [(0.0, 0.0), (float("inf"), float("inf")), (1, 1.0)],
        ids=["float_0", "float_inf", "int_1"],
    )
    def test_no_bounds(self, x_value: float, expected_output: float) -> None:
        output = check_real(x_value, "test")
        assert isinstance(output, float)
        assert output == expected_output

    def test_no_bounds_error(self) -> None:
        msg = "'test' should be a real number, but was of type <class 'complex'>"
        with pytest.raises(TypeError, match=msg):
            check_real(1 + 1j, "test")

    def test_l_bound(self) -> None:
        assert check_real(0, "test", l_bound=0) == 0
        assert check_real(-0, "test", l_bound=0) == 0

        msg = "'test' has an exclusive lower bound of 0, but was 0"
        with pytest.raises(ValueError, match=msg):
            check_real(0, "test", l_bound=0, l_inclusive=False)

        msg = "'test' has an inclusive lower bound of 0, but was -0.1"
        with pytest.raises(ValueError, match=msg):
            check_real(-0.1, "test", l_bound=0)

    def test_u_bound(self) -> None:
        assert check_real(0, "test", u_bound=0) == 0
        assert check_real(-0, "test", u_bound=0) == 0

        msg = "'test' has an exclusive upper bound of 0, but was 0"
        with pytest.raises(ValueError, match=msg):
            check_real(0, "test", u_bound=0, u_inclusive=False)

        msg = "'test' has an inclusive upper bound of 0, but was 0.1"
        with pytest.raises(ValueError, match=msg):
            check_real(0.1, "test", u_bound=0)


class TestCheckInt:
    def test_no_bounds(self) -> None:
        assert check_int(-1, "test") == -1
        assert check_int(1.0, "test") == 1

        msg = "'test' should be an integer, but was of type <class 'complex'>"
        with pytest.raises(TypeError, match=msg):
            check_int(1 + 1j, "test")

        msg = "'test' with value 1.1 could not be safely converted to an integer"
        with pytest.raises(ValueError, match=msg):
            check_int(1.1, "test")

    def test_l_bound(self) -> None:
        assert check_int(0, "test", l_bound=0) == 0
        assert check_int(-0, "test", l_bound=0) == 0

        msg = "'test' has an exclusive lower bound of 0, but was 0"
        with pytest.raises(ValueError, match=msg):
            check_int(0, "test", l_bound=0, l_inclusive=False)

        msg = "'test' has an inclusive lower bound of 0, but was -1"
        with pytest.raises(ValueError, match=msg):
            check_int(-1, "test", l_bound=0)

    def test_u_bound(self) -> None:
        assert check_int(0, "test", u_bound=0) == 0
        assert check_int(-0, "test", u_bound=0) == 0

        msg = "'test' has an exclusive upper bound of 0, but was 0"
        with pytest.raises(ValueError, match=msg):
            check_int(0, "test", u_bound=0, u_inclusive=False)

        msg = "'test' has an inclusive upper bound of 0, but was 1"
        with pytest.raises(ValueError, match=msg):
            check_int(1, "test", u_bound=0)


class TestCheckString:
    def test_no_keywords(self) -> None:
        assert check_string("Test", "test") == "Test"

    def test_lower(self) -> None:
        assert check_string("Test", "test", lower=True) == "test"

    def test_upper(self) -> None:
        assert check_string("Test", "test", upper=True) == "TEST"

    def test_error(self) -> None:
        msg = "'test' must be a string, but was of type <class 'int'>"
        with pytest.raises(TypeError, match=msg):
            check_string(1, "test")  # type: ignore[arg-type]


class TestCheckBool:
    def test_default(self) -> None:
        assert check_bool(True, "test")
        assert not check_bool(False, "test")

    def test_safe_is_false(self) -> None:
        assert check_bool(1, "test", safe=False)
        assert not check_bool(0, "test", safe=False)

    def test_safe_is_true(self) -> None:
        assert check_bool(True, "test", safe=True)
        with pytest.raises(TypeError):
            check_bool(0, "test", safe=True)


class TestAdjacencyMatrix:
    @pytest.mark.parametrize(
        "arg",
        [
            np.zeros((2, 2)),
            [[0, 0], [0, 0]],
            csr_matrix([[0, 0], [0, 0]]),
            ((0, 0), (0, 0)),
        ],
        ids=["ndarray", "nested_list", "csr_matrix", "nested_tuple"],
    )
    def test_check_adjacency_matrix_input(self, arg: ArrayLike) -> None:
        assert (check_adjacency_matrix(arg) == np.zeros((2, 2))).all()

    @pytest.mark.parametrize(
        "arg", [np.zeros(2), np.zeros((2, 3)), None], ids=["1d", "not_square", "None"]
    )
    def test_check_adjacency_matrix_errors(self, arg: Any) -> None:
        msg = "the provided value should be a square 2-D adjacency matrix"
        with pytest.raises(ValueError, match=msg):
            check_adjacency_matrix(arg)


class TestGraphValidTopology:
    def test_empty_graph(self) -> None:
        graph = nx.Graph()
        msg = "'test' has no nodes"
        with pytest.raises(ValueError, match=msg):
            check_graph_is_valid_topology(graph, "test")

    def test_line_graph(self) -> None:
        graph = nx.cycle_graph(2)
        check_graph_is_valid_topology(graph, "test")

    def test_self_loop(self) -> None:
        graph = nx.Graph()
        graph.add_edge(1, 1)
        msg = "'test' contains self-loops"
        with pytest.raises(ValueError, match=msg):
            check_graph_is_valid_topology(graph, "test")

    def test_non_int_label(self) -> None:
        graph = nx.Graph()
        graph.add_node((0, 0))

        msg = "'test' has nodes that are not integers"
        with pytest.raises(TypeError, match=msg):
            check_graph_is_valid_topology(graph, "test")


class TestCheckInstance:
    def test_default(self) -> None:
        check_instance(1, "test", int)
        check_instance(1, "test", Number)
        check_instance("TEST", "test", str)

    def test_error(self) -> None:
        msg = "'test' must an instance of <class 'str'>, but was of type <class 'int'>"
        with pytest.raises(TypeError, match=msg):
            check_instance(1, "test", str)


def test_warn_if_positive() -> None:
    warn_if_positive(0, "test")
    warn_if_positive(-1.0, "test")

    with pytest.warns(UserWarning) as record:
        warn_if_positive(1, "test")

    assert len(record) == 1
    assert isinstance(record[0].message, Warning)
    assert record[0].message.args[0] == "'test' was positive"


def test_warn_if_negative() -> None:
    warn_if_negative(0, "test")
    warn_if_negative(1.0, "test")

    with pytest.warns(UserWarning) as record:
        warn_if_negative(-1, "test")

    assert len(record) == 1
    assert isinstance(record[0].message, Warning)
    assert record[0].message.args[0] == "'test' was negative"
