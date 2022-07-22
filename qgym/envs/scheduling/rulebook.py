"""This module contains the CommutationRulebook class together with basic commutation
rules used in the scheduling environment.

Example:

# cnot gates with the same controll qubit commute
def cnot_commutation(gate1, gate2):
    if gate1[0] == "cnot" and gate2[0] == "cnot":
        if gate1[1] ==  gate2[1]:
            return True
    return False

rulebook = CommutionRulebook()
rulebook.add_rule(rulebook)
"""

from numbers import Integral
from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray


class CommutionRulebook:
    """Based on nearest blocking neighbour"""

    def __init__(self, default_rules: bool = True):
        """Init of the CommutionRulebook.

        :param default_rules: If True disjoint qubit and gates that are exactly the
            same commute. If False, then no rules will be initialized."""

        if default_rules:
            self.rulebook = [disjoint_qubits, same_gate]
        else:
            self.rulebook = []

    def make_blocking_matrix(
        self, circuit: List[Tuple[str, Integral, Integral]]
    ) -> NDArray[np.int_]:
        """Makes a 2xlen(circuit) array with dependencies based on the commutation given
        rules.

        :param circuit: circuit to check dependencies.
        :return: dependencies of the circuit bases on the rules and scheduling from
            right to left."""

        blocking_matrix = np.zeros((len(circuit), len(circuit)), dtype=bool)

        for idx, gate in enumerate(circuit):

            for idx_other in range(idx + 1, len(circuit)):

                gate_other = circuit[idx_other]

                if not self.commutes(gate, gate_other):
                    blocking_matrix[idx, idx_other] = True

        return blocking_matrix

    def commutes(
        self,
        gate1: Tuple[str, Integral, Integral],
        gate2: Tuple[str, Integral, Integral],
    ) -> bool:
        """Checks if gate1 and gate2 commute according to the rules in the rulebook.

        :param gate1: gate to check the commutation.
        :param gate2: gate to check gate1 against.
        :return: True if gate1 commutes with gate2. False
            otherwise."""
        for rule in self.rulebook:
            if rule(gate1, gate2):
                return True
        return False

    def add_rule(
        self,
        rule: Callable[
            [Tuple[str, Integral, Integral], Tuple[str, Integral, Integral]],
            bool,
        ],
    ) -> None:
        """Add a commutation rule to the rulebook

        :param rule: Rule to add to the rulebook. A rule takes as input two gates and
            returns True if two gate commutte according to the rule and False otherwise.
        """
        self.rulebook.append(rule)


def disjoint_qubits(
    gate1: Tuple[str, Integral, Integral], gate2: Tuple[str, Integral, Integral]
) -> bool:
    """Gates that have disjoint qubits commute.

    :param gate1: gate to check disjointness.
    :param gate2: gate to check disjointness against.
    :return: True if the gates are disjoint, False otherwise."""
    gate1_qubit1 = gate1[1]
    gate1_qubit2 = gate1[2]
    gate2_qubit1 = gate2[1]
    gate2_qubit2 = gate2[2]

    return (
        gate1_qubit1 != gate2_qubit1
        and gate1_qubit1 != gate2_qubit2
        and gate1_qubit2 != gate2_qubit1
        and gate1_qubit2 != gate2_qubit2
    )


def same_gate(
    gate1: Tuple[str, Integral, Integral], gate2: Tuple[str, Integral, Integral]
) -> bool:
    """Gates that have disjoint qubits commute.

    :param gate1: gate to check equality.
    :param gate2: gate to check equality against.
    :return: True if gate1 is equal to gate2, False otherwise."""
    if gate1 == gate2:
        return True
    return False
