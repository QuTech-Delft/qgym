"""
Rewarder for training an RL agent on the scheduling problem of OpenQL.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from qgym import Rewarder
from qgym.utils.input_validation import check_real, warn_if_negative, warn_if_positive


class BasicRewarder(Rewarder):
    """
    Basic rewarder for the Scheduling environment.
    """

    def __init__(
        self,
        illegal_action_penalty: float = -5.0,
        update_cycle_penalty: float = -1.0,
        schedule_gate_bonus: float = 0.0,
    ) -> None:
        """
        Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param update_cycle_penalty: penalty for updating the cycle.
        :param schedule_gate_bonus: bonus for scheduling a gate.
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
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the old_state and action.

        :param old_state: State of the Scheduling before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the Scheduling.
        :return: The reward for this action.
        """

        if action[1] != 0:
            return self._update_cycle_penalty

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        return self._schedule_gate_bonus

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: Dict[Any, Any]) -> bool:
        """
        Checks if the given action is illegal, i.e., checks if qubits are mapped
        multiple times.

        :param action: Action that has just been taken.
        :param old_state: State of the Scheduling before the current action.
        :return: Whether this action was illegal.
        """

        gate_to_schedule = action[0]
        return not old_state["legal_actions"][gate_to_schedule]

    def _set_reward_range(self) -> None:
        """
        Set the reward range.
        """
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
    """
    Episode rewarder for the Scheduling environment.
    """

    def __init__(
        self, illegal_action_penalty: float = -5.0, update_cycle_penalty: float = -1.0
    ) -> None:
        """
        Initialize the reward range and set the rewards and penalties.

        :param illegal_action_penalty: penalty for performing an illegal action.
        :param update_cycle_penalty: penalty for updating the cycle.
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
        old_state: Dict[Any, Any],
        action: NDArray[np.int_],
        new_state: Dict[Any, Any],
    ) -> float:
        """
        Compute a reward, based on the old_state and action.

        :param old_state: State of the Scheduling before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the Scheduling.
        :return: The reward for this action.
        """

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        if (new_state["schedule"] == -1).any():
            return 0

        reward = 0
        for gate_idx, scheduled_cycle in enumerate(new_state["schedule"]):
            gate = new_state["encoded_circuit"][gate_idx]
            finished = scheduled_cycle + new_state["gate_cycle_length"][gate.name]
            reward = min(reward, self._update_cycle_penalty * finished)

        return reward

    @staticmethod
    def _is_illegal(action: NDArray[np.int_], old_state: Dict[Any, Any]) -> bool:
        """
        Checks if the given action is illegal, i.e., checks if qubits are mapped
        multiple times.

        :param action: Action that has just been taken.
        :param old_state: State of the Scheduling before the current action.
        :return: Whether this action was illegal.
        """

        gate_to_schedule = action[0]
        return not old_state["legal_actions"][gate_to_schedule]

    def _set_reward_range(self) -> None:
        """
        Set the reward range.
        """
        l_bound = -float("inf")
        if self._illegal_action_penalty >= 0 and self._update_cycle_penalty >= 0:
            l_bound = 0

        u_bound = float("inf")
        if self._illegal_action_penalty <= 0 and self._update_cycle_penalty <= 0:
            u_bound = 0

        self._reward_range = (l_bound, u_bound)
