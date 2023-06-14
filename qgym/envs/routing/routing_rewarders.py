"""This module will contain some vanilla Rewarders for the ``Routing`` environment.

Usage:
    The rewarders in this module can be customized by initializing the rewarders with
    different values.

    .. code-block:: python

        from qgym.envs.routing import BasicRewarder

        rewarder = BasicRewarder(
            illegal_action_penalty = -1,
            update_cycle_penalty = -2,
            schedule_gate_bonus: = 3,
            )

    After initialization, the rewarders can be given to the ``Routing`` environment.

.. note::
    When implementing custom rewarders, they should inherit from ``qgym.Rewarder``.
    Furthermore, they must implement the ``compute_reward`` method. Which takes as input
    the old state, the new state and the given action. See the documentation of the
    ``qgym.envs.routing.routing`` module for more information on the state and
    action space.

"""
from abc import abstractmethod
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from qgym.envs.routing.routing_state import RoutingState
from qgym.templates import Rewarder
from qgym.utils.input_validation import check_real, warn_if_negative, warn_if_positive


class BasicRewarder(Rewarder):
    """RL Rewarder, for computing rewards on the ``RoutingState``."""

    def __init__(
        self,
        illegal_action_penalty: float = -50,
        penalty_per_swap: float = -10,
        reward_per_surpass: float = 10,
    ) -> None:
        self._illegal_action_penalty = check_real(
            illegal_action_penalty, "illegal_action_penalty"
        )
        self._penalty_per_swap = check_real(penalty_per_swap, "penalty_per_swap")
        self._reward_per_surpass = check_real(reward_per_surpass, "reward_per_surpass")
        self._set_reward_range()

        warn_if_positive(self._illegal_action_penalty, "illegal_action_penalty")
        warn_if_positive(self._penalty_per_swap, "penalty_per_swap")
        warn_if_negative(self._reward_per_surpass, "reward_per_surpass")

    def compute_reward(self, *, old_state: Any, action: Any) -> float:
        """Compute a reward, based on the old state, new state, and the given action.

        :param old_state: ``RoutingState`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: ``RoutingState`` after the current action.
        :return reward: The reward for this action.
        """

        if action[0] == 1 and old_state._is_legal_surpass(
            old_state.interaction_circuit[old_state.position][0],
            old_state.interaction_circuit[old_state.position][1],
        ):
            return self._reward_per_surpass
        elif action[0] == 0 and self._is_legal_swap(action[1], action[2]):
            return self._penalty_per_swap
        else:
            return self._illegal_action_penalty

    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if (
            self._illegal_action_penalty >= 0
            and self._penalty_per_swap >= 0
            and self._reward_per_surpass >= 0
        ):
            l_bound = 0

        u_bound = float("inf")
        if (
            self._illegal_action_penalty <= 0
            and self._penalty_per_swap <= 0
            and self._reward_per_surpass <= 0
        ):
            u_bound = 0

        self._reward_range = (l_bound, u_bound)

        def __eq__(self, other: Any) -> bool:
            return (
                type(self) is type(other)
                and self._reward_range == other._reward_range
                and self._illegal_action_penalty == other._illegal_action_penalty
                and self._reward_per_edge == other._reward_per_edge
                and self._penalty_per_edge == other._penalty_per_edge
            )

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Reward range of the rewarder. I.e., range that rewards can lie in."""
        return self._reward_range


class SwapQualityRewarder(BasicRewarder):
    """Rewarder for the ``InitialMapping`` environment, which gives a reward based on
    the improvement in the current step.
    """

    def __init__(
        self,
        illegal_action_penalty: float = -50,
        penalty_per_swap: float = -10,
        reward_per_surpass: float = 10,
        good_swap_reward: float = 10,
    ) -> None:
        self._illegal_action_penalty = check_real(
            illegal_action_penalty, "illegal_action_penalty"
        )
        self._penalty_per_swap = check_real(penalty_per_swap, "penalty_per_swap")
        self._reward_per_surpass = check_real(reward_per_surpass, "reward_per_surpass")
        self._good_swap_reward = check_real(good_swap_reward, "reward_per_surpass")
        self._set_reward_range()

        assert (
            0 <= self._good_swap_reward < -self._penalty_per_swap
        ), "Good swaps should not result in positive rewards."

        warn_if_positive(self._illegal_action_penalty, "illegal_action_penalty")
        warn_if_positive(self._penalty_per_swap, "penalty_per_swap")
        warn_if_negative(self._reward_per_surpass, "reward_per_surpass")
        warn_if_negative(self._good_swap_reward, "reward_per_good_swap")

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: NDArray[np.int_],
        new_state: RoutingState,
    ) -> float:
        """Compute a reward, based on the old state, the given action and the new state.
        Specifically .... are used

        :param old_state: ``RoutingState`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: ``RoutingState`` after the current action.
        :return reward: The reward for this action. If the action is illegal, then the
            reward is `illegal_action_penalty`. If the action is legal, then the reward
            for a surpass is just reward_per_surpass. But, for a legal swap the reward
            adjusted with respect to the BasicRewarder. Namely, the penalty of a swap is
            reduced if it increases the observation_reach and the penalty is increased
            if the observation_reach is decreases.
        """
        if action[0] == 1 and old_state._is_legal_surpass(
            old_state.interaction_circuit[old_state.position][0],
            old_state.interaction_circuit[old_state.position][1],
        ):
            return self._reward_per_surpass
        elif action[0] == 0 and self._is_legal_swap(action[1], action[2]):
            return (
                self._penalty_per_swap
                - self._good_swap_reward
                * self._observation_enhancement_factor(old_state, new_state)
            )
        else:
            return self._illegal_action_penalty

    def _observation_enhancement_factor(
        self,
        *,
        old_state: RoutingState,
        new_state: RoutingState,
    ) -> float:
        surpassing = True
        gate_number = 0
        while surpassing:
            if (
                new_state.obtain_observation["is_legal_surpass_booleans"][gate_number]
                == 1
            ):
                gate_number += 1
            else:
                surpassing = False
        new_direct_executable_gates_ahead = gate_number

        surpassing = True
        gate_number = 0
        while surpassing:
            if (
                old_state.obtain_observation["is_legal_surpass_booleans"][gate_number]
                == 1
            ):
                gate_number += 1
            else:
                surpassing = False
        old_direct_executable_gates_ahead = gate_number
        return (
            new_direct_executable_gates_ahead - old_direct_executable_gates_ahead
        ) / old_state.observation_reach


class EpisodeRewarder(BasicRewarder):
    """Rewarder for the ``Routing`` environment, which only gives a reward after at
    least N steps have been taken.
    """

    def compute_reward(
        self,
        *,
        N: int,
        new_state: RoutingState,
    ) -> float:
        """Compute a reward, based on the new state, and the given action. Specifically
        the connection graph, interaction graphs and mapping are used.

        :param N: number of steps over which the EpisodeRewarder determines the reward.
        :param new_state: ``RoutingState`` after the current action.
        :return reward: The reward calculated over the last N steps.
        """

        pass
