from copy import deepcopy
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from qgym.envs.initial_mapping.initial_mapping_rewarders import (
    BasicRewarder,
    EpisodeRewarder,
    SingleStepRewarder,
)
from qgym.rewarder import Rewarder


def _episode_generator(
    adjacency_matrices: Dict[str, Any]
) -> Iterator[Tuple[Dict[str, Any], NDArray[np.int_], Dict[str, Any]]]:

    old_state = deepcopy(adjacency_matrices)

    old_state["mapping_dict"] = {}
    old_state["logical_qubits_mapped"] = set()
    old_state["physical_qubits_mapped"] = set()

    new_state = deepcopy(old_state)

    action = np.array([0, 0])
    _perform_action(new_state, action)

    yield old_state, action, new_state

    for i in range(1, adjacency_matrices["connection_graph_matrix"].shape[0]):
        _perform_action(old_state, action)
        action = np.array([i, i])
        _perform_action(new_state, action)

        yield old_state, action, new_state


def _perform_action(state: Dict[str, Any], action: NDArray[np.int_]) -> None:
    state["mapping_dict"][action[0]] = action[1]
    state["logical_qubits_mapped"].add(action[0])
    state["physical_qubits_mapped"].add(action[1])


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder(), SingleStepRewarder(), EpisodeRewarder()]
)
def test_illegal_actions(rewarder: Rewarder) -> None:

    old_state = {"logical_qubits_mapped": {2}, "physical_qubits_mapped": {1}}
    action = np.array([1, 2])
    new_state = {}

    reward = rewarder.compute_reward(
        old_state=old_state, action=action, new_state=new_state
    )
    assert reward == -100


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder(), SingleStepRewarder(), EpisodeRewarder()]
)
def test_inheritance(rewarder):
    assert isinstance(rewarder, Rewarder)


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder, SingleStepRewarder, EpisodeRewarder]
)
def test_init(rewarder):
    rewarder = rewarder(-3, 3, -2)
    assert rewarder._reward_range == (-float("inf"), float("inf"))
    assert rewarder._illegal_action_penalty == -3
    assert rewarder._reward_per_edge == 3
    assert rewarder._penalty_per_edge == -2


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder, SingleStepRewarder, EpisodeRewarder]
)
@pytest.mark.parametrize(
    "illegal_action_penalty,reward_per_edge,penalty_per_edge,name",
    [
        ("test", 1, -1, "illegal_action_penalty"),
        (-1, "test", -1, "reward_per_edge"),
        (-1, 1, "test", "penalty_per_edge"),
    ],
)
def test_init_exceptions(
    rewarder, illegal_action_penalty, reward_per_edge, penalty_per_edge, name
):
    error_msg = f"'{name}' should be a real number, but was of type <class 'str'>"
    with pytest.raises(TypeError, match=error_msg):
        rewarder(illegal_action_penalty, reward_per_edge, penalty_per_edge)


@pytest.mark.parametrize(
    "rewarder", [BasicRewarder, SingleStepRewarder, EpisodeRewarder]
)
@pytest.mark.parametrize(
    "illegal_action_penalty,reward_per_edge,penalty_per_edge,msg",
    [
        (100, 1, -1, "'illegal_action_penalty' was postive"),
        (-1, -100, -1, "'reward_per_edge' was negative"),
        (-1, 1, 100, "'penalty_per_edge' was postive"),
    ],
)
def test_init_warnings(
    rewarder, illegal_action_penalty, reward_per_edge, penalty_per_edge, msg
):
    with pytest.warns(UserWarning) as record:
        rewarder(illegal_action_penalty, reward_per_edge, penalty_per_edge)

    assert len(record) == 1
    assert record[0].message.args[0] == msg


"""
Tests for the basic rewarder
"""

empty_graph = csr_matrix((3, 3))
full_graph = csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.parametrize(
    "adjacency_matrices,rewards",
    [
        (
            {
                "connection_graph_matrix": empty_graph,
                "interaction_graph_matrix": empty_graph,
            },
            [0, 0, 0],
        ),
        (
            {
                "connection_graph_matrix": full_graph,
                "interaction_graph_matrix": full_graph,
            },
            [0, 5, 15],
        ),
        (
            {
                "connection_graph_matrix": full_graph,
                "interaction_graph_matrix": empty_graph,
            },
            [0, 0, 0],
        ),
        (
            {
                "connection_graph_matrix": empty_graph,
                "interaction_graph_matrix": full_graph,
            },
            [0, -1, -3],
        ),
    ],
)
def test_basic_rewarder(
    adjacency_matrices: Dict[str, csr_matrix], rewards: List[float]
) -> None:
    episode_generator = _episode_generator(adjacency_matrices)

    rewarder = BasicRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]


"""
Tests for the single step rewarder
"""


@pytest.mark.parametrize(
    "adjacency_matrices,rewards",
    [
        (
            {
                "connection_graph_matrix": empty_graph,
                "interaction_graph_matrix": empty_graph,
            },
            [0, 0, 0],
        ),
        (
            {
                "connection_graph_matrix": full_graph,
                "interaction_graph_matrix": full_graph,
            },
            [0, 5, 10],
        ),
        (
            {
                "connection_graph_matrix": full_graph,
                "interaction_graph_matrix": empty_graph,
            },
            [0, 0, 0],
        ),
        (
            {
                "connection_graph_matrix": empty_graph,
                "interaction_graph_matrix": full_graph,
            },
            [0, -1, -2],
        ),
    ],
)
def test_single_step_rewarder(
    adjacency_matrices: Dict[str, csr_matrix], rewards: List[float]
) -> None:
    episode_generator = _episode_generator(adjacency_matrices)

    rewarder = SingleStepRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]


"""
Tests for the episode rewarder rewarder
"""


@pytest.mark.parametrize(
    "adjacency_matrices,rewards",
    [
        (
            {
                "connection_graph_matrix": empty_graph,
                "interaction_graph_matrix": empty_graph,
            },
            [0, 0, 0],
        ),
        (
            {
                "connection_graph_matrix": full_graph,
                "interaction_graph_matrix": full_graph,
            },
            [0, 0, 15],
        ),
        (
            {
                "connection_graph_matrix": full_graph,
                "interaction_graph_matrix": empty_graph,
            },
            [0, 0, 0],
        ),
        (
            {
                "connection_graph_matrix": empty_graph,
                "interaction_graph_matrix": full_graph,
            },
            [0, 0, -3],
        ),
    ],
)
def test_episode_step_rewarder(
    adjacency_matrices: Dict[str, csr_matrix], rewards: List[float]
) -> None:
    episode_generator = _episode_generator(adjacency_matrices)

    rewarder = EpisodeRewarder()

    for i, (old_state, action, new_state) in enumerate(episode_generator):
        reward = rewarder.compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        assert reward == rewards[i]
