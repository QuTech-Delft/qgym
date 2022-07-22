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

    def get_scheduled_after(
        self, circuit: List[Tuple[str, Integral, Integral]]
    ) -> NDArray[np.int_]:
        """Makes a 2xlen(circuit) array with dependencies based on the commutation given
        rules.

        :param circuit: circuit to check dependencies.
        :return: dependencies of the circuit bases on the rules and scheduling from
            right to left."""

        scheduled_after = np.zeros((2, len(circuit)), dtype=int)

        for idx, gate in enumerate(circuit):
            is_two_qubit_gate = gate[1] != gate[2]
            for idx_other in range(idx + 1, len(circuit)):

                gate_other = circuit[idx_other]

                if scheduled_after[0, idx] == 0:
                    if not self.commutes_on_first_qubit(gate, gate_other):
                        scheduled_after[0, idx] = idx_other

                if is_two_qubit_gate:
                    if scheduled_after[1, idx] == 0:
                        if not self.commutes_on_second_qubit(gate, gate_other):
                            scheduled_after[1, idx] = idx_other

        return scheduled_after

    def commutes_on_first_qubit(
        self,
        gate1: Tuple[str, Integral, Integral],
        gate2: Tuple[str, Integral, Integral],
    ) -> bool:
        """Checks if the first qubit of gate1 commutes with gate2

        :param gate1: gate to check the first qubit.
        :param gate2: gate to check gate1 against.
        :return: True if gate1 commutes with gate2 on the first qubit of gate1. False
            otherwise."""
        for rule in self.rulebook:
            if rule(gate1, gate2)[0]:
                return True
        return False

    def commutes_on_second_qubit(
        self,
        gate1: Tuple[str, Integral, Integral],
        gate2: Tuple[str, Integral, Integral],
    ) -> bool:
        """Checks if the second qubit of gate1 commutes with gate2

        :param gate1: gate to check the second qubit.
        :param gate2: gate to check gate1 against.
        :return: True if gate1 commutes with gate2 on the second qubit of gate1. False
            otherwise."""
        for rule in self.rulebook:
            if rule(gate1, gate2)[1]:
                return True
        return False

    def add_rules(
        self,
        rule: Callable[
            [Tuple[str, Integral, Integral], Tuple[str, Integral, Integral]],
            Tuple[bool, bool],
        ],
    ) -> None:
        """Add a commutation rule to the rulebook

        :param rule: Rule to add to the rulebook. A rule takes as input two gates and
            returns length 2 Boolean tuple stating if gate on commutes with gate2 on the
                first and second qubit respectively."""
        self.rulebook.append(rule)


def disjoint_qubits(
    gate1: Tuple[str, Integral, Integral], gate2: Tuple[str, Integral, Integral]
) -> Tuple[bool, bool]:
    """Gates that have disjoint qubits commute.

    :param gate1: gate to check disjointness.
    :param gate2: gate to check disjointness against.
    :return: (disjointness of first qubit, disjointness of second qubit)"""
    gate1_qubit1 = gate1[1]
    gate1_qubit2 = gate1[2]
    gate2_qubit1 = gate2[1]
    gate2_qubit2 = gate2[2]

    first_disjoint = gate1_qubit1 != gate2_qubit1 and gate1_qubit1 != gate2_qubit2
    second_disjoint = gate1_qubit2 != gate2_qubit1 and gate1_qubit2 != gate2_qubit2

    return first_disjoint, second_disjoint


def same_gate(
    gate1: Tuple[str, Integral, Integral], gate2: Tuple[str, Integral, Integral]
) -> Tuple[bool, bool]:
    """Gates that have disjoint qubits commute.

    :param gate1: gate to check equality.
    :param gate2: gate to check equality against.
    :return: (True, True) if gate1 is equal to gate2, (False, False) otherwise."""
    if gate1 == gate2:
        return True, True
    return False, False
