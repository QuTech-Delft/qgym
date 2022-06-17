"""
Rewarder for training an RL agent on the scheduling problem of OpenQL.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from qgym import Rewarder


class BasicRewarder(Rewarder):
    """
    Basic rewarder for the InitialMapping environment.
    """

    def __init__(
        self,
        illegal_action_penalty: float = -5,
        update_cycle_penalty: float = -1,
        schedule_gate_bonus: float = 1,
    ) -> None:
        """
        Initialize the reward range and set the rewards and penaltyies

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param update_cycle_penalty: penalty for updating the cycle
        :param schedule_gate_bonus: bonus for scheduling a gate
        """
        self._reward_range = (-float("inf"), float("inf"))
        self._illegal_action_penalty = illegal_action_penalty
        self._update_cycle_penalty = update_cycle_penalty
        self._schedule_gate_bonus = schedule_gate_bonus

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the old_state and action

        :param old_state: State of the Scheduling before the current action.
        :param action: Action that has just been taken
        :param new_state: Updated state of the Scheduling
        """
        reward = 0.0
        if action[1] != 0:
            reward += self._illegal_action_penalty

        if self._is_illegal(action, old_state):
            reward += self._illegal_action_penalty
        else:
            reward += self._schedule_gate_bonus

        return reward

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: Dict[Any, Any]) -> bool:
        """
        Checks if the given action is illegal, i.e. checks if qubits are mapped
        multiple times.

        :param action: Action that has just been taken
        :param old_state: State of the Scheduling before the current action.
        """

        gate_to_schedule = action[0]
        return not old_state["legal_actions"][gate_to_schedule]
