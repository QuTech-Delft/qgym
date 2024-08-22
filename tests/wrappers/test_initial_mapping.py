from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout, TrivialLayout, VF2Layout

from qgym.wrappers.initial_mapping import QiskitMapperWrapper


class TestQiskitMapperWrapper:

    @pytest.fixture(name="circuit")
    def circuit_fixture(self) -> QuantumCircuit:
        """Create a circuit with a cycle of cx gates."""
        circuit = QuantumCircuit(6)
        circuit.cx(list(range(6)), [*range(1, 6), 0])
        return circuit

    @pytest.fixture(name="coupling_map")
    def coupling_map_fixture(self) -> CouplingMap:
        """Create a coupling map for a 6 qubit device with a cycle topology."""
        return CouplingMap(zip(range(6), [*range(1, 6), 0]))

    def test_trivial_layout(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_mapper = TrivialLayout(coupling_map)
        mapper = QiskitMapperWrapper(qiskit_mapper)
        mapping = mapper.compute_mapping(circuit)
        np.testing.assert_equal(mapping, range(6))

    def test_vf2_layout(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_mapper = VF2Layout(coupling_map)
        mapper = QiskitMapperWrapper(qiskit_mapper)
        mapping = mapper.compute_mapping(circuit)
        optimal_mapping_found = False
        for i in range(6):
            opt_map1 = np.roll(np.arange(6), i)
            opt_map2 = np.flip(opt_map1)
            if np.array_equal(mapping, opt_map1) or np.array_equal(mapping, opt_map2):
                optimal_mapping_found = True
                break
        assert optimal_mapping_found

    def test_sabre_layout(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_mapper = SabreLayout(coupling_map)
        mapper = QiskitMapperWrapper(qiskit_mapper)
        mapper.compute_mapping(circuit)
