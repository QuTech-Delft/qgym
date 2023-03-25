"""This module contains dataclasses used in the ``Routing`` environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
from numpy.typing import NDArray

from qgym.custom_types import Gate
from qgym.utils.gate_encoder import GateEncoder
from qgym.utils.random_circuit_generator import RandomCircuitGenerator


@dataclass
class RoutingUtils:
    """Utils used in the ``Routing`` environment."""

    random_circuit_generator: RandomCircuitGenerator
    random_circuit_mode: str
    gate_encoder: GateEncoder


@dataclass
class CircuitInfo:
    """Info of the circuit of ``Routing`` environment."""

    # TODO: make the circuit info relevant for routing?!
    # TODO: check whether this dataclass functions in stripped down version

    encoded: List[Gate]
    names: NDArray[np.int_]
    acts_on: NDArray[np.int_]
    legal: NDArray[np.bool_]
    dependencies: NDArray[np.int_]

    def reset(self, circuit: Optional[List[Gate]], utils: RoutingUtils) -> CircuitInfo:
        """Reset the object.

        To be used in the reset function of the ``Scheduling`` environment.

        :returns: Self.
        """
        if circuit is None:
            circuit = utils.random_circuit_generator.generate_circuit(
                mode=utils.random_circuit_mode
            )

        self.encoded = utils.gate_encoder.encode_gates(circuit)
        return self
