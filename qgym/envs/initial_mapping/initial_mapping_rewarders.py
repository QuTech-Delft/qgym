"""
Environment and rewarder for training an RL agent on the initial mapping problem of
OpenQL.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from qgym import Rewarder


class BasicRewarder(Rewarder):
    """
    Basic rewarder for the InitialMapping environment.
    """

    def __init__(
        self,
        illegal_action_penalty: Optional[float] = -100,
        reward_per_edge: Optional[float] = 5,
        penalty_per_edge: Optional[float] = -1,
    ) -> None:
        """
        Initialize the reward range and set the rewards and penalties

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param reward_per_edge: reward for performing a 'good' action
        :param penalty_per_edge: penalty for performing a 'bad' action
        """
        self._reward_range = (-float("inf"), float("inf"))
        self._illegal_action_penalty = illegal_action_penalty
        self._reward_per_edge = reward_per_edge
        self._penalty_per_edge = penalty_per_edge

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the current state, and the connection and
        interaction graphs.

        :param old_state: State of the InitialMapping before the current action.
        :param action: Action that has just been taken
        :param new_state: Updated state of the InitialMapping
        :return reward: The reward for this action.
        """

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._compute_state_reward(new_state)

    def _compute_state_reward(self, state: Dict[Any, Any]) -> float:
        """
        Compute the value of the mapping defined by the input state.

        :param state: The state to compute the value of.
        :return: Value of this state.
        """
        reward = 0.0
        for interaction_i, interaction_j in zip(
            *state["interaction_graph_matrix"].nonzero()
        ):
            mapped_interaction_i = state["mapping_dict"].get(interaction_i, None)
            mapped_interaction_j = state["mapping_dict"].get(interaction_j, None)
            if mapped_interaction_i is None or mapped_interaction_j is None:
                continue
            if (
                state["connection_graph_matrix"][
                    mapped_interaction_i, mapped_interaction_j
                ]
                == 0
            ):
                reward += self._penalty_per_edge
            else:
                reward += self._reward_per_edge

        return reward / 2  # divide by two due to double counting of edges

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: Dict[Any, Any]) -> bool:
        """
        Checks if the given action is illegal i.e., checks if qubits are mapped
        multiple times.

        :param action: Action that has just been taken
        :param old_state: State of the InitialMapping before the current action.
        :return: Whether this action is valid for the given state.
        """
        return (
            action[0] in old_state["physical_qubits_mapped"]
            or action[1] in old_state["logical_qubits_mapped"]
        )

    def __eq__(self, o) -> bool:
        return (
            type(self) == type(o)
            and self._reward_range == o._reward_range
            and self._illegal_action_penalty == o._illegal_action_penalty
            and self._reward_per_edge
            and o._reward_per_edge
            and self._penalty_per_edge == o._penalty_per_edge
        )


class SingleStepRewarder(BasicRewarder):
    """
    Rewarder for the InitialMapping environment, which gives a reward based on
    the improvement in the current step.
    """

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the current state, and the connection and
        interaction graphs.

        :param old_state: State of the InitialMapping before the current action.
        :param action: Action that has just been taken
        :param new_state: Updated state of the InitialMapping
        :return reward: The reward for this action.
        """

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._compute_state_reward(new_state) - self._compute_state_reward(
            old_state
        )


class EpisodeRewarder(BasicRewarder):
    """
    Rewarder for the InitialMapping environment, which only gives a reward at
    the end of the episode or when an illegal action is taken.
    """

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the current state, and the connection and
        interaction graphs.

        :param old_state: State of the InitialMapping before the current action.
        :param action: Action that has just been taken
        :param new_state: Updated state of the InitialMapping
        :return reward: The reward for this action.
        """

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        if (
            len(new_state["physical_qubits_mapped"])
            != new_state["connection_graph_matrix"].shape[0]
        ):
            return 0

        return self._compute_state_reward(new_state)
