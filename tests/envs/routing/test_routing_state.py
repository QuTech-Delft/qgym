"""This module contains tests for the ``RoutingState`` class."""
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import qgym.spaces
from qgym.envs.routing.routing_state import RoutingState


# Arrange
@pytest.fixture(name="quad_graph", scope="class")
def quad_graph_fixture() -> nx.Graph:
    quad_graph = nx.Graph()
    quad_graph.add_edge(0, 1)
    quad_graph.add_edge(1, 2)
    quad_graph.add_edge(2, 3)
    quad_graph.add_edge(3, 0)
    return quad_graph


@pytest.fixture(name="simple_state")
def simple_state_fixture(quad_graph: nx.Graph) -> RoutingState:
    return RoutingState(
        max_interaction_gates=50,
        max_observation_reach=5,
        connection_graph=quad_graph,
        observe_legal_surpasses=False,
        observe_connection_graph=True,
    )


@pytest.mark.parametrize(
    "max_interaction_gates, max_observation_reach",
    [(1, 1), (5, 5), (10, 5), (50, 5), (5, 10)],
    scope="class",
)
def test_init(
    max_interaction_gates: int, max_observation_reach: int, quad_graph: nx.Graph
) -> None:
    state = RoutingState(
        max_interaction_gates=max_interaction_gates,
        max_observation_reach=max_observation_reach,
        connection_graph=quad_graph,
        observe_legal_surpasses=False,
        observe_connection_graph=True,
    )
    assert state.max_interaction_gates == max_interaction_gates
    assert isinstance(state.swap_gates_inserted, Iterable)
    assert state.position == 0
    assert state.steps_done == 0
    circuit = np.asarray(state.interaction_circuit)
    assert circuit.shape[1] == 2
    assert (circuit < state.n_qubits).all()
    assert state.connection_graph is quad_graph


def test_create_observation_space(simple_state: RoutingState) -> None:
    observation_space = simple_state.create_observation_space()
    assert isinstance(observation_space, qgym.spaces.Dict)
    assert isinstance(
        observation_space["interaction_gates_ahead"], qgym.spaces.MultiDiscrete
    )
    assert isinstance(observation_space["mapping"], qgym.spaces.MultiDiscrete)


def test_interaction_circuit_properties(simple_state: RoutingState) -> None:
    assert len(simple_state.interaction_circuit) <= simple_state.max_interaction_gates
    assert isinstance(simple_state.interaction_circuit, np.ndarray)


class TestCanBeExecuted:
    @pytest.mark.parametrize("qubit1, qubit2", [(0, 3), (0, 1), (1, 2), (2, 3)])
    def test_succes(self, simple_state: RoutingState, qubit1: int, qubit2: int) -> None:
        assert simple_state.is_legal_surpass(qubit1, qubit2)
        assert simple_state.is_legal_surpass(qubit2, qubit1)

    @pytest.mark.parametrize("qubit1, qubit2", [(1, 3), (0, 2)])
    def test_fail(self, simple_state: RoutingState, qubit1: int, qubit2: int) -> None:
        assert not simple_state.is_legal_surpass(qubit1, qubit2)
        assert not simple_state.is_legal_surpass(qubit2, qubit1)


def test_obtain_observation(simple_state: RoutingState) -> None:
    observation = simple_state.obtain_observation()
    observation_space = simple_state.create_observation_space()
    assert observation in observation_space


class TestUpdateState:
    @pytest.mark.parametrize(
        argnames="action, expected_mapping",
        argvalues=[([0, 2, 3], [0, 1, 3, 2]), ([0, 1, 3], [0, 1, 2, 3])],
        ids=["legal", "illegal"],
    )
    def test_swap(
        self,
        simple_state: RoutingState,
        action: NDArray[np.int_],
        expected_mapping: ArrayLike,
    ) -> None:
        simple_state.update_state(np.asarray(action))
        assert simple_state.position == 0
        assert simple_state.steps_done == 1
        assert np.array_equal(simple_state.mapping, expected_mapping)

    @pytest.mark.parametrize(
        argnames="interaction_circuit, expected_position",
        argvalues=[([(0, 1)], 1), ([(0, 2)], 0)],
        ids=["legal", "illegal"],
    )
    def test_surpass(
        self,
        simple_state: RoutingState,
        interaction_circuit: List[Tuple[int, int]],
        expected_position: int,
    ) -> None:
        simple_state.interaction_circuit = interaction_circuit
        simple_state.update_state(np.array([1, 10, 10]))
        assert simple_state.position == expected_position
        assert simple_state.steps_done == 1
        assert np.array_equal(simple_state.mapping, [0, 1, 2, 3])


def test_reset(simple_state: RoutingState) -> None:
    assert isinstance(simple_state.reset(), RoutingState)
    assert len(simple_state.swap_gates_inserted) == 0
    assert simple_state.position == 0
    assert simple_state.steps_done == 0
