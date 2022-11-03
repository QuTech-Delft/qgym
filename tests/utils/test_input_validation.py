from numbers import Number

import networkx as nx
import numpy as np
import pytest
from qgym.utils.input_validation import (
    check_adjacency_matrix,
    check_graph_is_valid_topology,
    check_instance,
    check_int,
    check_real,
    check_string,
    warn_if_negative,
    warn_if_positive,
)
from scipy.sparse import csr_matrix


def test_check_real_no_bounds():
    assert check_real(float("inf"), "test") == float("inf")

    msg = "'test' should be a real number, but was of type <class 'complex'>"
    with pytest.raises(TypeError, match=msg):
        check_real(1 + 1j, "test")


def test_check_real_l_bound():
    assert check_real(0, "test", l_bound=0) == 0
    assert check_real(-0, "test", l_bound=0) == 0

    msg = "'test' has an exclusive lower bound of 0, but was 0"
    with pytest.raises(ValueError, match=msg):
        check_real(0, "test", l_bound=0, l_inclusive=False)

    msg = "'test' has an inclusive lower bound of 0, but was -0.1"
    with pytest.raises(ValueError, match=msg):
        check_real(-0.1, "test", l_bound=0)


def test_check_real_u_bound():
    assert check_real(0, "test", u_bound=0) == 0
    assert check_real(-0, "test", u_bound=0) == 0

    msg = "'test' has an exclusive upper bound of 0, but was 0"
    with pytest.raises(ValueError, match=msg):
        check_real(0, "test", u_bound=0, u_inclusive=False)

    msg = "'test' has an inclusive upper bound of 0, but was 0.1"
    with pytest.raises(ValueError, match=msg):
        check_real(0.1, "test", u_bound=0)


def test_check_int_no_bounds():
    assert check_int(-1, "test") == -1
    assert check_int(1.0, "test") == 1

    msg = "'test' should be an integer, but was of type <class 'complex'>"
    with pytest.raises(TypeError, match=msg):
        check_int(1 + 1j, "test")

    msg = "'test' with value 1.1 could not be safely converted to an integer"
    with pytest.raises(ValueError, match=msg):
        check_int(1.1, "test")


def test_check_int_l_bound():
    assert check_int(0, "test", l_bound=0) == 0
    assert check_int(-0, "test", l_bound=0) == 0

    msg = "'test' has an exclusive lower bound of 0, but was 0"
    with pytest.raises(ValueError, match=msg):
        check_int(0, "test", l_bound=0, l_inclusive=False)

    msg = "'test' has an inclusive lower bound of 0, but was -1"
    with pytest.raises(ValueError, match=msg):
        check_int(-1, "test", l_bound=0)


def test_check_int_u_bound():
    assert check_int(0, "test", u_bound=0) == 0
    assert check_int(-0, "test", u_bound=0) == 0

    msg = "'test' has an exclusive upper bound of 0, but was 0"
    with pytest.raises(ValueError, match=msg):
        check_int(0, "test", u_bound=0, u_inclusive=False)

    msg = "'test' has an inclusive upper bound of 0, but was 1"
    with pytest.raises(ValueError, match=msg):
        check_int(1, "test", u_bound=0)


def test_check_string():
    assert check_string("Test", "test") == "Test"
    assert check_string("Test", "test", lower=True) == "test"
    assert check_string("Test", "test", upper=True) == "TEST"

    msg = "'test' must be a string, but was of type <class 'int'>"
    with pytest.raises(TypeError, match=msg):
        check_string(1, "test")


@pytest.mark.parametrize(
    "arg",
    [
        np.zeros((2, 2)),
        [[0, 0], [0, 0]],
        csr_matrix([[0, 0], [0, 0]]),
        ((0, 0), (0, 0)),
    ],
)
def test_check_adjacency_matrix_input(arg):
    assert (check_adjacency_matrix(arg) == np.zeros((2, 2))).all()


@pytest.mark.parametrize(
    "arg",
    [np.zeros(2), np.zeros((2, 3))],
)
def test_check_adjacency_matrix_errors(arg):
    msg = "The provided value should be a square 2-D adjacency matrix."
    with pytest.raises(ValueError, match=msg):
        check_adjacency_matrix(arg)


def test_check_graph_is_valid_topology():
    graph = nx.Graph()
    msg = "'test' has no nodes"
    with pytest.raises(ValueError, match=msg):
        check_graph_is_valid_topology(graph, "test")

    graph.add_edge(1, 2)
    check_graph_is_valid_topology(graph, "test")

    graph.add_edge(1, 1)
    msg = "'test' contains self-loops"
    with pytest.raises(ValueError, match=msg):
        check_graph_is_valid_topology(graph, "test")


def test_check_instance():
    check_instance(1, "test", int)
    check_instance(1, "test", Number)

    msg = "'test' must an instance of <class 'str'>, but was of type <class 'int'>"
    with pytest.raises(TypeError, match=msg):
        check_instance(1, "test", str)


def test_warn_if_positive():
    warn_if_positive(0, "test")
    warn_if_positive(-1.0, "test")

    with pytest.warns(UserWarning) as record:
        warn_if_positive(1, "test")

    assert len(record) == 1
    assert record[0].message.args[0] == "'test' was positive"


def test_warn_if_negative():
    warn_if_negative(0, "test")
    warn_if_negative(1.0, "test")

    with pytest.warns(UserWarning) as record:
        warn_if_negative(-1, "test")

    assert len(record) == 1
    assert record[0].message.args[0] == "'test' was negative"
