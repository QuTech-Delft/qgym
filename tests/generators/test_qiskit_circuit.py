import networkx as nx
import pytest
from qiskit import QuantumCircuit

from qgym.generators import MaxCutQAOAGenerator


class TestMaxCutQAOAGenerator:

    @pytest.mark.parametrize("n_nodes", range(1, 6))
    def test_iterating(self, n_nodes: int) -> None:
        generator = MaxCutQAOAGenerator(n_nodes, 0.5, seed=42)
        for circuit, _ in zip(generator, range(10)):
            assert isinstance(circuit, QuantumCircuit)
            assert circuit.num_qubits == n_nodes
            assert circuit.num_parameters == 0
            ops = circuit.count_ops()
            assert not (set(ops) - {"barrier", "h", "rzz", "measure", "rx"})
            assert ops["h"] == n_nodes
            assert ops["measure"] == n_nodes
            assert ops["rx"] == n_nodes
            if "rzz" in ops:
                assert ops["rzz"] <= n_nodes * (n_nodes - 1) / 2

    @pytest.mark.parametrize("n_layers", range(1, 6))
    def test_maxcut_qaoa_circuit(self, n_layers: int) -> None:
        generator = MaxCutQAOAGenerator(5, 0.5, n_layers=n_layers)
        circuit = generator._maxcut_qaoa_circuit(nx.cycle_graph(5))
        assert circuit.num_qubits == 5
        assert circuit.num_parameters == 0
        ops = circuit.count_ops()
        assert not (set(ops) - {"barrier", "h", "rzz", "measure", "rx"})
        assert ops["h"] == 5
        assert ops["measure"] == 5
        assert ops["rx"] == n_layers * 5
        assert ops["rzz"] == n_layers * 5
