from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import BasicSwap, SabreSwap
from stable_baselines3 import PPO

from qgym.envs.routing import BasicRewarder, Routing
from qgym.wrappers.routing import AgentRoutingWrapper, QiskitRoutingWrapper


class TestQiskitRoutingWrapper:
    @pytest.fixture(name="circuit")
    def circuit_fixture(self) -> QuantumCircuit:
        """Create a circuit with a cycle of cx gates."""
        circuit = QuantumCircuit(6)
        circuit.cx(list(range(6)), [*range(1, 6), 0])
        return circuit

    @pytest.fixture(name="coupling_map")
    def coupling_map_fixture(self) -> CouplingMap:
        """Create a coupling map for a 6 qubit device with a line topology."""
        return CouplingMap(zip(range(5), range(1, 6)))

    def test_basic_swap(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_router = BasicSwap(coupling_map)
        router = QiskitRoutingWrapper(qiskit_router)
        routed_dag = router.compute_routing(circuit)

        expected_circuit = QuantumCircuit(6)
        expected_circuit.cx(range(5), range(1, 6))
        expected_circuit.swap(range(5, 1, -1), range(4, 0, -1))
        expected_circuit.cx(1, 0)

        assert expected_circuit == dag_to_circuit(routed_dag)

    def test_sabre_swap(
        self, circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> None:
        qiskit_router = SabreSwap(coupling_map, seed=42)
        router = QiskitRoutingWrapper(qiskit_router)
        routed_dag = router.compute_routing(circuit)

        expected_circuit = QuantumCircuit(6)
        expected_circuit.cx(range(5), range(1, 6))
        expected_circuit.swap([0, 1, 4, 3], [1, 2, 5, 4])
        expected_circuit.cx(3, 2)

        assert expected_circuit == dag_to_circuit(routed_dag)


class TestAgentMapperWrapper:
    @pytest.fixture(name="circuit")
    def circuit_fixture(self) -> QuantumCircuit:
        """Create a circuit with a cycle of cx gates."""
        circuit = QuantumCircuit(3)
        circuit.cx([0, 1, 2], [1, 2, 0])
        return circuit

    @pytest.fixture(name="env")
    def env_fixture(self) -> Routing:
        connection_graph = nx.from_edgelist([(0, 1), (1, 2)])
        rewarder = BasicRewarder(-2, -1, 3)
        env = Routing(
            connection_graph=connection_graph,
            max_observation_reach=3,
            rewarder=rewarder,
        )
        env.rng = np.random.default_rng(42)
        return env

    @pytest.fixture(name="agent")
    def agent_fixture(self, env: Routing) -> PPO:
        agent = PPO("MultiInputPolicy", env, seed=42)
        agent.learn(1000)
        return agent

    def test_wrapper(self, agent: PPO, env: Routing, circuit: QuantumCircuit) -> None:
        router = AgentRoutingWrapper(agent, env, 100)
        routed_dag = router.compute_routing(circuit)
        properties = routed_dag.properties()
        operations = properties["operations"]
        assert operations["cx"] == 3
        assert operations["swap"] >= 1

        layout = Layout.generate_trivial_layout(routed_dag.qregs["q"])
        for gate in routed_dag.two_qubit_ops():
            qubit1 = layout[gate.qargs[0]]
            qubit2 = layout[gate.qargs[1]]
            if {qubit1, qubit2} not in [{0, 1}, {1, 2}]:
                breakpoint()
