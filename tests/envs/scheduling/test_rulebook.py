import numpy as np
import pytest

from qgym.custom_types import Gate
from qgym.envs.scheduling.rulebook import (
    CommutationRulebook,
    disjoint_qubits,
    same_gate,
)


@pytest.mark.parametrize(
    "default_rules, rules",
    [(False, []), (True, [disjoint_qubits, same_gate])],
)
def test_init(default_rules, rules):
    rulebook = CommutationRulebook(default_rules=default_rules)
    assert rulebook._rules == rules


@pytest.fixture
def default_rulebook():
    return CommutationRulebook()


@pytest.mark.parametrize(
    "gate1, gate2, commutes",
    [
        (Gate("x", 1, 1), Gate("x", 1, 1), True),
        (Gate("x", 1, 1), Gate("x", 2, 2), True),
        (Gate("x", 1, 1), Gate("y", 1, 1), False),
        (Gate("x", 1, 1), Gate("y", 2, 2), True),
        (Gate("cnot", 1, 2), Gate("cnot", 1, 2), True),
        (Gate("cnot", 1, 2), Gate("cnot", 1, 3), False),
        (Gate("cnot", 4, 2), Gate("cnot", 1, 3), True),
    ],
)
def test_commutes(default_rulebook, gate1, gate2, commutes):
    assert default_rulebook.commutes(gate1, gate2) == commutes
    assert default_rulebook.commutes(gate2, gate1) == commutes


@pytest.mark.parametrize(
    "circuit, expected_matrix",
    [
        ([Gate("x", 1, 1)], np.array([[0]])),
        ([Gate("x", 1, 1), Gate("y", 2, 2)], np.array([[0, 0], [0, 0]])),
        ([Gate("x", 1, 1), Gate("y", 1, 1)], np.array([[0, 1], [0, 0]])),
        (
            [Gate("x", 1, 1), Gate("cnot", 1, 2), Gate("y", 2, 2)],
            np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        ),
        (
            [Gate("x", 1, 1), Gate("cnot", 1, 2), Gate("y", 1, 1)],
            np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        ),
    ],
)
def test_make_blocking_matrix(default_rulebook, circuit, expected_matrix):
    blocking_matrix = default_rulebook.make_blocking_matrix(circuit)
    assert np.array_equal(blocking_matrix, expected_matrix)


def test_add_rule(default_rulebook):
    def always_commute(gate_1, gate2):
        return True

    default_rulebook.add_rule(always_commute)
    assert default_rulebook._rules[-1] == always_commute
    assert default_rulebook.commutes(Gate("x", 1, 1), Gate("y", 1, 1))
