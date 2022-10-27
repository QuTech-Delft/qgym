"""
This module contains the CommutationRulebook class together with basic commutation
rules used in the scheduling environment.

Example::

    from qgym.env.scheduling.rulebook import CommutationRulebook

    # cnot gates with the same control qubit commute
    def cnot_commutation(gate1, gate2):
        if gate1[0] == "cnot" and gate2[0] == "cnot":
            if gate1[1] ==  gate2[1]:
                return True
        return False

    # init the rulebook and add the commutation rule
    rulebook = CommutationRulebook()
    rulebook.add_rule(rulebook)

"""

from typing import Callable, List

import numpy as np
from numpy.typing import NDArray
from qgym.custom_types import Gate


class CommutationRulebook:
    """
    Based on nearest blocking neighbour
    """

    def __init__(self, default_rules: bool = True) -> None:
        """
        Init of the CommutationRulebook.

        :param default_rules: If True, default rules are used. Default rules dictate
            that gates with disjoint qubits commute and that gates that are the
            same commute. If False, then no rules will be initialized.
        """
        if default_rules:
            self._rules = [disjoint_qubits, same_gate]
        else:
            self._rules = []

    def make_blocking_matrix(self, circuit: List[Gate]) -> NDArray[np.int_]:
        """
        Makes a len(circuit) x len(circuit) array with dependencies based on the given
        commutation rules.

        :param circuit: circuit to check dependencies.
        :return: dependencies of the circuit bases on the rules and scheduling from
            right to left.
        """

        blocking_matrix = np.zeros((len(circuit), len(circuit)), dtype=bool)

        for idx, gate in enumerate(circuit):
            for idx_other in range(idx + 1, len(circuit)):

                gate_other = circuit[idx_other]

                if not self.commutes(gate, gate_other):
                    blocking_matrix[idx, idx_other] = True

        return blocking_matrix

    def commutes(self, gate1: Gate, gate2: Gate) -> bool:
        """
        Checks if gate1 and gate2 commute according to the rules in the rulebook.

        :param gate1: gate to check the commutation.
        :param gate2: gate to check gate1 against.
        :return: Whether gate1 commutes with gate2.
        """
        for rule in self._rules:
            if rule(gate1, gate2):
                return True
        return False

    def add_rule(self, rule: Callable[[Gate, Gate], bool]) -> None:
        """
        Add a commutation rule to the rulebook

        :param rule: Rule to add to the rulebook. A rule takes as input two gates and
        :return: Whether the two gates commute according to the given rule.
        """
        self._rules.append(rule)


def disjoint_qubits(gate1: Gate, gate2: Gate) -> bool:
    """
    Gates that have disjoint qubits commute.

    :param gate1: gate to check disjointness.
    :param gate2: gate to check disjointness against.
    :return: Whether the gates are disjoint.
    """
    return (
        gate1.q1 != gate2.q1
        and gate1.q1 != gate2.q2
        and gate1.q2 != gate2.q1
        and gate1.q2 != gate2.q2
    )


def same_gate(gate1: Gate, gate2: Gate) -> bool:
    """
    Gates that are equal commute.

    :param gate1: gate to check equality.
    :param gate2: gate to check equality against.
    :return: Whether the gates are equal.
    """
    return gate1 == gate2
