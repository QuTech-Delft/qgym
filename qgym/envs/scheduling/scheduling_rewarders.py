"""
Rewarder for training an RL agent on the scheduling problem of OpenQL.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict

import numpy as np
from numpy.random._examples.cffi.extending import state
from numpy.typing import NDArray

from qgym import Rewarder


class BasicRewarder(Rewarder):
    """
    Basic rewarder for the Scheduling environment.
    """

    def __init__(
        self,
        illegal_action_penalty: Real = -5.0,
        update_cycle_penalty: Real = -1.0,
        schedule_gate_bonus: Real = 0.0,
    ) -> None:
        """
        Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param update_cycle_penalty: penalty for updating the cycle.
        :param schedule_gate_bonus: bonus for scheduling a gate.
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
        Compute a reward, based on the old_state and action.

        :param old_state: State of the Scheduling before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the Scheduling.
        """

        reward = 0.0
        if action[1] != 0:
            reward += self._update_cycle_penalty

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

        :param action: Action that has just been taken.
        :param old_state: State of the Scheduling before the current action.
        """

        gate_to_schedule = action[0]
        return not old_state["legal_actions"][gate_to_schedule]


class EpisodeRewarder(Rewarder):
    def __init__(
        self,
        illegal_action_penalty: Real = -5.0,
        cycle_used_penalty: Real = -1.0
    ) -> None:
        """
        Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param update_cycle_penalty: penalty for updating the cycle.
        :param schedule_gate_bonus: bonus for scheduling a gate.
        """

        self._reward_range = (-float("inf"), 0)
        self._illegal_action_penalty = float(illegal_action_penalty)
        self._cycle_used_penalty = float(cycle_used_penalty)

    def compute_reward(
        self,
        *,
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the old_state and action.

        :param old_state: State of the Scheduling before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the Scheduling.
        """
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty
        elif bool((new_state["schedule"] == -1).any()):
            return 0
        reward = 0
        for gate_idx, scheduled_cycle in enumerate(new_state["schedule"]):
            gate = new_state["encoded_circuit"][gate_idx]
            finished = scheduled_cycle + new_state["gate_cycle_length"][gate.name]
            reward = min(reward, self._cycle_used_penalty * finished)
        return reward

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: Dict[Any, Any]) -> bool:
        """
        Checks if the given action is illegal, i.e. checks if qubits are mapped
        multiple times.

        :param action: Action that has just been taken.
        :param old_state: State of the Scheduling before the current action.
        """

        gate_to_schedule = action[0]
        return not old_state["legal_actions"][gate_to_schedule]

