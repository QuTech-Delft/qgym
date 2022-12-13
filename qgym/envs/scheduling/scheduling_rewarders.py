"""This module contains some vanilla Rewarders for the ``Scheduling`` environment.

Usage:
    The rewarders in this module can be customized by initializing the rewarders with
    different values.

    .. code-block:: python

        from qgym.envs.scheduling import BasicRewarder

        rewarder = BasicRewarder(
            illegal_action_penalty = -1,
            update_cycle_penalty = -2,
            schedule_gate_bonus: = 3,
            )

    After initialization, the rewarders can be given to the ``Scheduling`` environment.

.. note::
    When implementing custom rewarders, they should inherit from ``qgym.Rewarder``.
    Furthermore, they must implement the ``compute_reward`` method. Which takes as input
    the old state, the new state and the given action. See the documentation of the
    ``qgym.envs.scheduling.scheduling`` module for more information on the state and
    action space.

"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from qgym import Rewarder
from qgym.envs.scheduling.scheduling_state import SchedulingState
from qgym.utils.input_validation import check_real, warn_if_negative, warn_if_positive


class BasicRewarder(Rewarder):
    """Basic rewarder for the ``Scheduling`` environment."""

    def __init__(
        self,
        illegal_action_penalty: float = -5.0,
        update_cycle_penalty: float = -1.0,
        schedule_gate_bonus: float = 0.0,
    ) -> None:
        """Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: Penalty for performing an illegal action. An
            action is illegal if ``action[0]`` is not in ``state["legal_actions"]``.
            This value should be negative (but is not required) and defaults to -5.
        :param update_cycle_penalty: Penalty given for incrementing a cycle. Since the
            ``Scheduling`` environment wats to create the shortest schedules,
            incrementing the cycle should be penalized. This value should
            be negative (but is not required) and defaults to -1.
        :param schedule_gate_bonus: Reward gained for successfully scheduling a gate.
            This value should be positive (but is not required) and defaults to 0.
        """
        self._illegal_action_penalty = check_real(
            illegal_action_penalty, "illegal_action_penalty"
        )
        self._update_cycle_penalty = check_real(
            update_cycle_penalty, "update_cycle_penalty"
        )
        self._schedule_gate_bonus = check_real(
            schedule_gate_bonus, "schedule_gate_bonus"
        )
        self._set_reward_range()

        warn_if_positive(self._illegal_action_penalty, "illegal_action_penalty")
        warn_if_positive(self._update_cycle_penalty, "update_cycle_penalty")
        warn_if_negative(self._schedule_gate_bonus, "schedule_gate_bonus")

    def compute_reward(
        self,
        *,
        old_state: SchedulingState,
        action: NDArray[np.int_],
        new_state: SchedulingState,
    ) -> float:
        """Compute a reward, based on the new state, and the given action. Specifically
        the 'legal_actions' actions array.

        :param old_state: State of the ``Scheduling`` environment before the current
            action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``Scheduling`` environment.
        :return reward: The reward for this action. If the action is illegal, then the
            reward is `illegal_action_penalty`. If the action is legal, and increments
            the cycle, then the reward is `update_cycle_penalty`. Otherwise, the reward
            is `schedule_gate_bonus`.
        """
        if action[1] != 0:
            return self._update_cycle_penalty

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._schedule_gate_bonus

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: SchedulingState) -> bool:
        """Check if the given action is illegal. An action is illegal if ``action[0]``
        is not in ``old_state["legal_actions"]``.

        :param action: Action that has just been taken.
        :param old_state: State of the ``Scheduling`` before the current action.
        :return: Boolean value stating whether this action was illegal.
        """
        gate_to_schedule = action[0]
        return not old_state.circuit_info.legal[gate_to_schedule]

    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if (
            self._illegal_action_penalty >= 0
            and self._update_cycle_penalty >= 0
            and self._schedule_gate_bonus >= 0
        ):
            l_bound = 0

        u_bound = float("inf")
        if (
            self._illegal_action_penalty <= 0
            and self._update_cycle_penalty <= 0
            and self._schedule_gate_bonus <= 0
        ):
            u_bound = 0

        self._reward_range = (l_bound, u_bound)


class EpisodeRewarder(Rewarder):
    """Rewarder for the ``Scheduling`` environment, which only gives a reward at
    the end of the episode or when an illegal action is taken.
    """

    def __init__(
        self, illegal_action_penalty: float = -5.0, update_cycle_penalty: float = -1.0
    ) -> None:
        """Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: Penalty for performing an illegal action. An
            action is illegal if ``action[0]`` is not in ``state["legal_actions"]``.
            This value should be negative (but is not required) and defaults to -5.
        :param update_cycle_penalty: Penalty given for incrementing a cycle. Since the
            ``Scheduling`` environment wats to create the shortest schedules,
            incrementing the cycle should be penalized. This value should
            be negative (but is not required) and defaults to -1.
        """
        self._illegal_action_penalty = check_real(
            illegal_action_penalty, "illegal_action_penalty"
        )
        self._update_cycle_penalty = check_real(
            update_cycle_penalty, "update_cycle_penalty"
        )
        self._set_reward_range()

        warn_if_positive(self._illegal_action_penalty, "illegal_action_penalty")
        warn_if_positive(self._update_cycle_penalty, "update_cycle_penalty")

    def compute_reward(
        self,
        *,
        old_state: SchedulingState,
        action: NDArray[np.int_],
        new_state: SchedulingState,
    ) -> float:
        """Compute a reward, based on the new state, and the given action.

        :param old_state: State of the ``Scheduling`` environment before the current
            action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``Scheduling`` environment.
        :return reward: The reward for this action. If the action is illegal, then the
            reward is `illegal_action_penalty`. If the action is legal, but the episode
            is not yet done, then the reward is 0. Otherwise, the reward is
            `update_cycle_penalty`x`current cycle`.
        """
        if action[1] == 0 and self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        if not new_state.is_done():
            return 0

        reward = 0
        for gate_idx, scheduled_cycle in enumerate(new_state.circuit_info.schedule):
            gate = new_state.circuit_info.encoded[gate_idx]
            finished = scheduled_cycle + new_state.gates[gate.name].cycle_length
            reward = min(reward, self._update_cycle_penalty * finished)

        return reward

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: SchedulingState) -> bool:
        """Check if the given action is illegal. An action is illegal if ``action[0]``
        is not in ``old_state["legal_actions"]``.

        :param action: Action that has just been taken.
        :param old_state: State of the ``Scheduling`` before the current action.
        :return: Boolean value stating whether this action was illegal.
        """
        gate_to_schedule = action[0]
        return not old_state.circuit_info.legal[gate_to_schedule]

    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if self._illegal_action_penalty >= 0 and self._update_cycle_penalty >= 0:
            l_bound = 0

        u_bound = float("inf")
        if self._illegal_action_penalty <= 0 and self._update_cycle_penalty <= 0:
            u_bound = 0

        self._reward_range = (l_bound, u_bound)
