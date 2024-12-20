from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
import pytest

from qgym.envs.routing import (
    BasicRewarder,
    EpisodeRewarder,
    RoutingState,
    SwapQualityRewarder,
)
from qgym.generators.interaction import NullInteractionGenerator
from qgym.templates.rewarder import Rewarder

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import ArrayLike


def _episode_generator(
    interaction_circuit: ArrayLike,
) -> Iterator[tuple[RoutingState, int, RoutingState]]:
    interaction_circuit = np.asarray(interaction_circuit)

    connection_graph = nx.Graph()
    connection_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    new_state = RoutingState(
        interaction_generator=NullInteractionGenerator(),
        max_observation_reach=5,
        connection_graph=connection_graph,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
    )
    new_state.reset(interaction_circuit=interaction_circuit)

    for _ in range(100):
        if new_state.is_done():
            break

        next_gate = new_state.interaction_circuit[new_state.position]
        if new_state.is_legal_surpass(*next_gate):
            action = new_state.n_connections
        else:
            physical_qubit1 = new_state.mapping[next_gate[0]]
            if physical_qubit1 in (0, 1):
                action = new_state.edges.index((2, 3))
            else:
                action = new_state.edges.index((0, 1))

        old_state = deepcopy(new_state)
        new_state.update_state(action)
        yield old_state, action, new_state


@pytest.fixture(
    name="rewarder",
    params=(
        BasicRewarder(),
        SwapQualityRewarder(),
        EpisodeRewarder(),
    ),
)
def rewarder_fixture(request: pytest.FixtureRequest) -> Rewarder:
    return cast("Rewarder", request.param)


def test_illegal_actions(rewarder: Rewarder) -> None:
    episode_generator = _episode_generator([[0, 1], [0, 2]])
    _, _, new_state = next(episode_generator)
    old_state = deepcopy(new_state)

    reward = rewarder.compute_reward(old_state=old_state, action=4, new_state=new_state)
    assert reward == -50


def test_inheritance(rewarder: Rewarder) -> None:
    assert isinstance(rewarder, Rewarder)


@pytest.mark.parametrize(
    ("circuit", "rewards"),
    [
        ([(0, 1), (1, 2), (2, 3), (3, 0)], [3, 6, 9, 12]),
        ([(0, 1), (0, 2), (1, 3)], [3, -7, -4, -1]),
    ],
    ids=["no-swap", "1-swap"],
)
def test_basic_rewarder(circuit: ArrayLike, rewards: Iterable[float]) -> None:
    episode_generator = _episode_generator(circuit)
    rewarder = BasicRewarder(
        illegal_action_penalty=-100, penalty_per_swap=-10, reward_per_surpass=3
    )
    for (old_state, action, new_state), reward in zip(episode_generator, rewards):
        computed_reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )
        assert computed_reward == reward


@pytest.mark.parametrize(
    ("circuit", "rewards"),
    [
        ([(0, 1), (1, 2), (2, 3), (3, 0)], [0, 0, 0, 0]),
        ([(0, 1), (0, 2), (1, 3)], [0, 0, 0, -10]),
    ],
    ids=["no-swap", "1-swap"],
)
def test_episode_rewarder(circuit: ArrayLike, rewards: Iterable[float]) -> None:
    episode_generator = _episode_generator(circuit)
    rewarder = EpisodeRewarder(illegal_action_penalty=-100, penalty_per_swap=-10)
    for (old_state, action, new_state), reward in zip(episode_generator, rewards):
        computed_reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )
        assert computed_reward == reward


@pytest.mark.parametrize(
    ("circuit", "rewards"),
    [
        ([(0, 1), (1, 2), (2, 3), (3, 0)], [3, 3, 3, 3]),
        ([(0, 1), (0, 2), (1, 3)], [3, -8, 3, 3]),
    ],
    ids=["no-swap", "1-swap"],
)
def test_swap_quality_rewarder(circuit: ArrayLike, rewards: Iterable[float]) -> None:
    episode_generator = _episode_generator(circuit)
    rewarder = SwapQualityRewarder(
        illegal_action_penalty=-100,
        penalty_per_swap=-10,
        reward_per_surpass=3,
        good_swap_reward=5,
    )
    for (old_state, action, new_state), reward in zip(episode_generator, rewards):
        computed_reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )
        assert computed_reward == reward


def test_swap_quality_rewarder_error() -> None:
    circuit = [(0, 2)]
    episode_generator = _episode_generator(circuit)
    rewarder = SwapQualityRewarder()

    old_state, action, new_state = next(episode_generator)
    old_state.observe_legal_surpasses = False
    with pytest.raises(
        ValueError, match="observe_legal_surpasses needs to be True to compute"
    ):
        rewarder.compute_reward(old_state=old_state, action=action, new_state=new_state)
