"""This module contains some vanilla Rewarders for the ``InitialMapping`` environment.

Usage:
    The rewarders in this module can be customized by initializing the rewarders with
    different values.

    .. code-block:: python

        from qgym.envs.initial_mapping.initial_mapping_rewarders import BasicRewarder

        rewarder = BasicRewarder(
            illegal_action_penalty = -1,
            reward_per_edge = 5,
            penalty_per_edge: = -2,
            )

    After initialization, the rewarders can be given to the ``InitialMapping``
    environment.

.. note::
    When implementing custom rewarders, they should inherit from ``qgym.Rewarder``.
    Furthermore, they must implement the ``compute_reward`` method. Which takes as input
    the old state, the new state and the given action. See the documentation of the
    ``qgym.envs.initial_mapping.initial_mapping`` module for more information on
    the state and action space.

"""
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from qgym.utils.input_validation import check_real, warn_if_negative, warn_if_positive

from qgym import Rewarder


class BasicRewarder(Rewarder):
    """Basic rewarder for the ``InitialMapping`` environment."""

    def __init__(
        self,
        illegal_action_penalty: Optional[float] = -100,
        reward_per_edge: Optional[float] = 5,
        penalty_per_edge: Optional[float] = -1,
    ) -> None:
        """Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: Penalty for performing an illegal action. An
            action is illegal if the action contains a virtual or physical qubit that
            has already been mapped. This value should be negative (but is not required)
            and defaults to -100.
        :param reward_per_edge: Reward gained per 'good' edge in the interaction graph.
            An edge is 'good' if the mapped edge overlaps with an edge of the
            connection graph. This value should be positive (but is not required) and
            defaults to 5.
        :param penalty_per_edge: Penalty given per 'bad' edge in the interaction graph.
            An edge is 'bad' if the edge is mapped and is not 'good'. This value should
            be negative (but is not required) and defaults to -1.
        """
        self._illegal_action_penalty = check_real(
            illegal_action_penalty, "illegal_action_penalty"
        )
        self._reward_per_edge = check_real(reward_per_edge, "reward_per_edge")
        self._penalty_per_edge = check_real(penalty_per_edge, "penalty_per_edge")
        self._set_reward_range()

        warn_if_positive(self._illegal_action_penalty, "illegal_action_penalty")
        warn_if_negative(self._reward_per_edge, "reward_per_edge")
        warn_if_positive(self._penalty_per_edge, "penalty_per_edge")

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """Compute a reward, based on the new state, and the given action. Specifically
        the connection graph, interaction graphs and mapping are used.

        :param old_state: State of the ``InitialMapping`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``InitialMapping``.
        :return reward: The reward for this action. If the action is illegal, then the
            reward is `illegal_action_penalty`. If the action is legal, then the reward
            is the *total* number of 'good' edges times `reward_per_edge` plus the
            *total* number of 'bad' edges times `penalty_per_edge`.
        """
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._compute_state_reward(new_state)

    def _compute_state_reward(self, state: Dict[Any, Any]) -> float:
        """Compute the value of the mapping defined by the input state.

        :param state: The state to compute the value of.
        :return: The reward value of this state.
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
        """Check if the given action is illegal i.e., checks if qubits are mapped
        multiple times.

        :param action: Action that has just been taken.
        :param old_state: State of the ``InitialMapping`` before the current action.
        :return: Whether this action is valid for the given state.
        """
        return (
            action[0] in old_state["physical_qubits_mapped"]
            or action[1] in old_state["logical_qubits_mapped"]
        )

    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if (
            self._reward_per_edge >= 0
            and self._penalty_per_edge >= 0
            and self._illegal_action_penalty >= 0
        ):
            l_bound = 0

        u_bound = float("inf")
        if (
            self._reward_per_edge <= 0
            and self._penalty_per_edge <= 0
            and self._illegal_action_penalty <= 0
        ):
            u_bound = 0

        self._reward_range = (l_bound, u_bound)

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) == type(other)
            and self._reward_range == other._reward_range
            and self._illegal_action_penalty == other._illegal_action_penalty
            and self._reward_per_edge == other._reward_per_edge
            and self._penalty_per_edge == other._penalty_per_edge
        )


class SingleStepRewarder(BasicRewarder):
    """Rewarder for the ``InitialMapping`` environment, which gives a reward based on
    the improvement in the current step.
    """

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """Compute a reward, based on the new state, and the given action. Specifically
        the connection graph, interaction graphs and mapping are used.

        :param old_state: State of the ``InitialMapping`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``InitialMapping``.
        :return reward: The reward for this action. If the action is illegal, then the
            reward is `illegal_action_penalty`. If the action is legal, then the reward
            is the number of 'good' edges times `reward_per_edge` plus the number of
            'bad' edges times `penalty_per_edge` created by the *this* action.
        """
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._compute_state_reward(new_state) - self._compute_state_reward(
            old_state
        )


class EpisodeRewarder(BasicRewarder):
    """Rewarder for the ``InitialMapping`` environment, which only gives a reward at
    the end of the episode or when an illegal action is taken.
    """

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """Compute a reward, based on the new state, and the given action. Specifically
        the connection graph, interaction graphs and mapping are used.

        :param old_state: State of the ``InitialMapping`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``InitialMapping``.
        :return reward: The reward for this action. If the action is illegal, then the
            reward is `illegal_action_penalty`. If the action is legal, but the mapping
            is not yet finished, then the reward is 0. If the action is legal, and the
            mapping is finished, then the reward is the number of 'good' edges times
            `reward_per_edge` plus the number of 'bad' edges times `penalty_per_edge`.
        """
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        if (
            len(new_state["physical_qubits_mapped"])
            != new_state["connection_graph_matrix"].shape[0]
        ):
            return 0

        return self._compute_state_reward(new_state)
