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


class BasicRewarder:
    """RL Rewarder, for computing rewards on the ``RoutingState``."""

    def __init__(
    self,
    illegal_surpass_penalty: float = -50,
    illegal_swap_penalty: float = -50,
    penalty_per_swap: float = -10,
    reward_per_surpass: float = 10,
    ) -> None:
        self._illegal_surpass_penalty = check_real(
            illegal_surpass_penalty, "illegal_surpass_penalty"
        )
        self._illegal_swap_penalty = check_real(
            illegal_swap_penalty, "illegal_swap_penalty"
        )
        self._penalty_per_swap = check_real(penalty_per_swap, "penalty_per_swap")
        self._reward_per_surpass = check_real(reward_per_surpass, "reward_per_surpass")
        self._set_reward_range()

        warn_if_positive(self._illegal_surpass_penalty, "illegal_surpass_penalty")
        warn_if_positive(self._illegal_swap_penalty, "illegal_swap_penalty")
        warn_if_positive(self._penalty_per_swap, "penalty_per_swap")
        warn_if_negative(self._reward_per_surpass, "reward_per_surpass")

    @abstractmethod
    def compute_reward(self, *, old_state: Any, action: Any, new_state: Any) -> float:
        """Compute a reward, based on the old state, new state, and the given action.

        :param old_state: ``RoutingState`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: ``RoutingState`` after the current action.
        :return reward: The reward for this action.
        """
        old_state.
        raise NotImplementedError
        # TODO: implement compute reward functionality
    
    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if (
            self._illegal_surpass_penalty >= 0
            and self._illegal_swap_penalty >= 0
            and self._penalty_per_swap >= 0
            and self._reward_per_surpass >= 0
        ):
            l_bound = 0

        u_bound = float("inf")
        if (
            self._illegal_surpass_penalty <= 0
            and self._illegal_swap_penalty <= 0
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

class SingleStepRewarder(BasicRewarder):
    """Rewarder for the ``InitialMapping`` environment, which gives a reward based on
    the improvement in the current step.
    """

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
            is the number of 'good' edges times `reward_per_edge` plus the number of
            'bad' edges times `penalty_per_edge` created by the *this* action.
        """
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._compute_state_reward(new_state) - self._compute_state_reward(
            old_state
        )


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