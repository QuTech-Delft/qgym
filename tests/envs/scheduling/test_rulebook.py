from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

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
def test_init(default_rules: bool, rules: list[Callable[[Gate, Gate], bool]]) -> None:
    rulebook = CommutationRulebook(default_rules=default_rules)
    assert rulebook._rules == rules


@pytest.fixture
def default_rulebook() -> CommutationRulebook:
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
def test_commutes(
    default_rulebook: CommutationRulebook, gate1: Gate, gate2: Gate, commutes: bool
) -> None:
    assert default_rulebook.commutes(gate1, gate2) == commutes
    assert default_rulebook.commutes(gate2, gate1) == commutes


@pytest.mark.parametrize(
    "circuit, expected_matrix",
    [
        ([Gate("x", 1, 1)], [[0]]),
        ([Gate("x", 1, 1), Gate("y", 2, 2)], [[0, 0], [0, 0]]),
        ([Gate("x", 1, 1), Gate("y", 1, 1)], [[0, 1], [0, 0]]),
        (
            [Gate("x", 1, 1), Gate("cnot", 1, 2), Gate("y", 2, 2)],
            [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        ),
        (
            [Gate("x", 1, 1), Gate("cnot", 1, 2), Gate("y", 1, 1)],
            [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
        ),
    ],
)
def test_make_blocking_matrix(
    default_rulebook: CommutationRulebook,
    circuit: list[Gate],
    expected_matrix: ArrayLike,
) -> None:
    blocking_matrix = default_rulebook.make_blocking_matrix(circuit)
    np.testing.assert_array_equal(blocking_matrix, expected_matrix)


def test_add_rule(default_rulebook: CommutationRulebook) -> None:
    def always_commute(gate_1: Gate, gate2: Gate) -> bool:
        return True

    default_rulebook.add_rule(always_commute)
    assert default_rulebook._rules[-1] == always_commute
    assert default_rulebook.commutes(Gate("x", 1, 1), Gate("y", 1, 1))
