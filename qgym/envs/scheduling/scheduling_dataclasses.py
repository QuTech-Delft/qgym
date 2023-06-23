"""This module contains dataclasses used in the ``Scheduling`` environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
from numpy.typing import NDArray

from qgym.custom_types import Gate
from qgym.envs.scheduling.rulebook import CommutationRulebook
from qgym.utils.gate_encoder import GateEncoder
from qgym.utils.random_circuit_generator import RandomCircuitGenerator


@dataclass
class SchedulingUtils:
    """Utils used in the ``Scheduling`` environment."""

    random_circuit_generator: RandomCircuitGenerator
    random_circuit_mode: str
    rulebook: CommutationRulebook
    gate_encoder: GateEncoder


@dataclass
class GateInfo:
    """Info of a specific gate used in the ``Scheduling`` environment."""

    cycle_length: int
    not_in_same_cycle: Set[int]
    exclude: int = 0
    exclude_next_cycle: bool = False

    def reset(self) -> GateInfo:
        """Reset the object.

        To be used in the reset function of the ``Scheduling`` environment.

        :returns: Self.
        """
        self.exclude = 0
        self.exclude_next_cycle = False
        return self


@dataclass
class CircuitInfo:
    """Info of the circuit of the current episode of ``Scheduling`` environment."""

    encoded: List[Gate]
    names: NDArray[np.int_]
    acts_on: NDArray[np.int_]
    legal: NDArray[np.int8]
    dependencies: NDArray[np.int_]
    schedule: NDArray[np.int_]
    blocking_matrix: NDArray[np.int_]

    def reset(
        self, circuit: Optional[List[Gate]], utils: SchedulingUtils
    ) -> CircuitInfo:
        """Reset the object.

        To be used in the reset function of the ``Scheduling`` environment.

        :returns: Self.
        """
        if circuit is None:
            circuit = utils.random_circuit_generator.generate_circuit(
                mode=utils.random_circuit_mode
            )

        self.blocking_matrix = utils.rulebook.make_blocking_matrix(circuit)
        self.encoded = utils.gate_encoder.encode_gates(circuit)
        self.schedule = np.full(len(circuit), -1, dtype=int)
        return self
