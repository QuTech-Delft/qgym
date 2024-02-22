"""This module contains tests for the ``RoutingState`` class."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import pytest

import qgym.spaces
from qgym.envs.routing.routing_state import RoutingState
from qgym.generators.interaction import InteractionGenerator, NullInteractionGenerator


# Arrange
@pytest.fixture(name="quad_graph", scope="class")
def quad_graph_fixture() -> nx.Graph:
    quad_graph = nx.Graph()
    quad_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return quad_graph


@pytest.fixture(name="simple_state")
def simple_state_fixture(quad_graph: nx.Graph) -> RoutingState:
    return RoutingState(
        interaction_generator=NullInteractionGenerator(),
        max_observation_reach=5,
        connection_graph=quad_graph,
        observe_legal_surpasses=False,
        observe_connection_graph=True,
    )


@pytest.mark.parametrize("max_observation_reach", [1, 5, 10, 50])
def test_init(max_observation_reach: int, quad_graph: nx.Graph) -> None:
    state = RoutingState(
        interaction_generator=NullInteractionGenerator(),
        max_observation_reach=max_observation_reach,
        connection_graph=quad_graph,
        observe_legal_surpasses=False,
        observe_connection_graph=True,
    )
    assert isinstance(state.interaction_generator, InteractionGenerator)
    assert isinstance(state.swap_gates_inserted, Iterable)
    assert state.position == 0
    assert state.steps_done == 0
    circuit = np.asarray(state.interaction_circuit)
    assert circuit.shape[1] == 2
    assert (circuit < state.n_qubits).all()
    assert state.connection_graph is quad_graph


class TestCreateObservationSpace:
    def test_min_case(self, quad_graph: nx.Graph) -> None:
        state = RoutingState(
            interaction_generator=NullInteractionGenerator(),
            max_observation_reach=5,
            connection_graph=quad_graph,
            observe_legal_surpasses=False,
            observe_connection_graph=False,
        )
        observation_space = state.create_observation_space()
        assert isinstance(observation_space, qgym.spaces.Dict)
        assert isinstance(
            observation_space["interaction_gates_ahead"], qgym.spaces.MultiDiscrete
        )
        assert isinstance(observation_space["mapping"], qgym.spaces.MultiDiscrete)

    def test_observe_legal_surpasses(self, quad_graph: nx.Graph) -> None:
        state = RoutingState(
            interaction_generator=NullInteractionGenerator(),
            max_observation_reach=5,
            connection_graph=quad_graph,
            observe_legal_surpasses=True,
            observe_connection_graph=False,
        )
        observation_space = state.create_observation_space()
        assert isinstance(observation_space, qgym.spaces.Dict)
        assert isinstance(
            observation_space["interaction_gates_ahead"], qgym.spaces.MultiDiscrete
        )
        assert isinstance(observation_space["mapping"], qgym.spaces.MultiDiscrete)
        assert isinstance(
            observation_space["is_legal_surpass"], qgym.spaces.MultiBinary
        )

    def test_connection_graph(self, quad_graph: nx.Graph) -> None:
        state = RoutingState(
            interaction_generator=NullInteractionGenerator(),
            max_observation_reach=5,
            connection_graph=quad_graph,
            observe_legal_surpasses=False,
            observe_connection_graph=True,
        )
        observation_space = state.create_observation_space()
        assert isinstance(observation_space, qgym.spaces.Dict)
        assert isinstance(
            observation_space["interaction_gates_ahead"], qgym.spaces.MultiDiscrete
        )
        assert isinstance(observation_space["mapping"], qgym.spaces.MultiDiscrete)
        assert isinstance(
            observation_space["connection_graph"], qgym.spaces.MultiBinary
        )


def test_interaction_circuit_properties(simple_state: RoutingState) -> None:
    assert isinstance(simple_state.interaction_circuit, np.ndarray)
    assert simple_state.interaction_circuit.ndim == 2
    assert simple_state.interaction_circuit.shape[1] == 2


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
    def test_swap(
        self,
        simple_state: RoutingState,
    ) -> None:
        action = simple_state.edges.index((2, 3))
        expected_mapping = [0, 1, 3, 2]
        simple_state.update_state(action)
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
        interaction_circuit: list[tuple[int, int]],
        expected_position: int,
    ) -> None:
        simple_state.interaction_circuit = np.array(interaction_circuit)
        simple_state.update_state(4)
        assert simple_state.position == expected_position
        assert simple_state.steps_done == 1
        assert np.array_equal(simple_state.mapping, [0, 1, 2, 3])


def test_reset(simple_state: RoutingState) -> None:
    assert isinstance(simple_state.reset(), RoutingState)
    assert len(simple_state.swap_gates_inserted) == 0
    assert simple_state.position == 0
    assert simple_state.steps_done == 0
