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
import warnings

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
        """Set the rewards and penalties.

        :param illegal_action_penalty: Penalty for performing an illegal action. An
            action is illegal if ``action[0]`` is not in ``state["legal_actions"]``.
            This value should be negative (but is not required) and defaults to -50.
        :param penalty_per_swap: Penalty for placing a swap. In general, we want to have
            as little swaps as possible. Therefore, this value should be negative and
            defaults to -10.
        :param reward_per_surpass: Reward given for surpassing a gate. In general, we
            want to have go to the end of the circuit as fast as possible. Therefore,
            this value should be positive and defaults to 10.
        """
        self._illegal_action_penalty = check_real(
            illegal_action_penalty, "illegal_action_penalty"
        )
        self._penalty_per_swap = check_real(penalty_per_swap, "penalty_per_swap")
        self._reward_per_surpass = check_real(reward_per_surpass, "reward_per_surpass")
        self._set_reward_range()

        warn_if_positive(self._illegal_action_penalty, "illegal_action_penalty")
        warn_if_positive(self._penalty_per_swap, "penalty_per_swap")
        warn_if_negative(self._reward_per_surpass, "reward_per_surpass")

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: NDArray[np.int_],
        new_state: RoutingState,
    ) -> float:
        """Compute a reward, based on the old state, new state, and the given action.

        :param old_state: ``RoutingState`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: ``RoutingState`` after the current action.
        :returns reward: The reward for this action.
        """

        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        reward = old_state.position * self._reward_per_surpass
        reward += len(old_state.swap_gates_inserted) * self._penalty_per_swap
        reward += self._reward_per_surpass if action[0] else self._penalty_per_swap

        return reward

    def _is_illegal(self, action: NDArray[np.int_], old_state: RoutingState) -> bool:
        """Checks whether an action chosen by the agent is illegal.
        returns: a boolean.
        """
        if action[0]:
            qubit1, qubit2 = old_state.interaction_circuit[old_state.position]
            return not old_state.is_legal_surpass(qubit1, qubit2)

        return not old_state.is_legal_swap(action[1], action[2])

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


class SwapQualityRewarder(BasicRewarder):
    """Rewarder for the ``Routing`` environment, which has an adjusted reward w.r.t.
    the BasicRewarder in the sense that good SWAPs give lower penalties and bad SWAPs
    give higher penalties.
    """

    def __init__(
        self,
        illegal_action_penalty: float = -50,
        penalty_per_swap: float = -10,
        reward_per_surpass: float = 10,
        good_swap_reward: float = 5,
    ) -> None:
        """Set the rewards and penalties and a flag.

        :param illegal_action_penalty: Penalty for performing an illegal action. An
            action is illegal if ``action[0]`` is not in ``state["legal_actions"]``.
            This value should be negative (but is not required) and defaults to -50.
        :param penalty_per_swap: Penalty for placing a swap. In general, we want to have
            as little swaps as possible. Therefore, this value should be negative and
            defaults to -10.
        :param reward_per_surpass: Reward given for surpassing a gate. In general, we
            want to have go to the end of the circuit as fast as possible. Therefore,
            this value should be positive and defaults to 10.
        :param good_swap_reward: Reward given for placing a good swap. In general, we
            want to place as little swaps as possible. However, when they are good, the
            penalty for the placement should be surpressed. That happens with this
            reward. So, the value should be positive and smaller than the
            penalty_per_swap, in order not to get positive rewards for swaps, default=5.
        """
        super().__init__(
            illegal_action_penalty=illegal_action_penalty,
            penalty_per_swap=penalty_per_swap,
            reward_per_surpass=reward_per_surpass,
        )

        self._good_swap_reward = check_real(good_swap_reward, "reward_per_good_swap")

        if not 0 <= self._good_swap_reward < -self._penalty_per_swap:
            warnings.warn("Good swaps should not result in positive rewards.")

        warn_if_negative(self._good_swap_reward, "reward_per_good_swap")

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: NDArray[np.int_],
        new_state: RoutingState,
    ) -> float:
        """Compute a reward, based on the old state, the given action and the new state.
        Specifically, the change in observation reach is used.

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
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        if action[0]:
            return self._reward_per_surpass

        return (
            self._penalty_per_swap
            + self._good_swap_reward
            * self._observation_enhancement_factor(old_state, new_state)
        )

    def _observation_enhancement_factor(
        self,
        old_state: RoutingState,
        new_state: RoutingState,
    ) -> float:
        """Calculates the change of the observation reach as an effect of a swap.

        :param old_state: ``RoutingState`` before the current action.
        :param new_state: ``RoutingState`` after the current action.
        :return float: A fraction that expresses the procentual improvement w.r.t the
            old_state's observation.
        """
        try:
            is_legal_surpass = old_state.obtain_observation()["is_legal_surpass"]
            old_executable_gates_ahead = int(is_legal_surpass.sum())

            is_legal_surpass = new_state.obtain_observation()["is_legal_surpass"]
            new_executable_gates_ahead = int(is_legal_surpass.sum())
        except KeyError as error:
            if not old_state.observe_legal_surpasses:
                msg = "observe_legal_surpasses needs to be True to compute"
                msg += "observation_enhancement_factor"
                raise ValueError(msg) from error
            raise error

        return (
            new_executable_gates_ahead - old_executable_gates_ahead
        ) / old_state.max_observation_reach

    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if (
            self._illegal_action_penalty >= 0
            and self._penalty_per_swap >= 0
            and self._reward_per_surpass >= 0
            and self._good_swap_reward >= 0
        ):
            l_bound = 0

        u_bound = float("inf")
        if (
            self._illegal_action_penalty <= 0
            and self._penalty_per_swap <= 0
            and self._reward_per_surpass <= 0
            and self._good_swap_reward <= 0
        ):
            u_bound = 0

        self._reward_range = (l_bound, u_bound)


class EpisodeRewarder(BasicRewarder):
    """Rewarder for the ``Routing`` environment, which only gives a reward after at
    the end of a full episode. The reward is the lowest for the lowest amount of SWAPs.
    This could be improved for setting for taking into account the fidelity of edges and
    scoring good and looking at what edges the circuit is executed.
    """

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: NDArray[np.int_],
        new_state: RoutingState,
    ) -> float:
        """Compute a reward, based on the new state, and the given action.

        :param new_state: ``RoutingState`` after the current action.
        :return reward: The reward calculated over the episode.
        """
        if self._is_illegal(action, old_state):
            return self._illegal_action_penalty

        if not new_state.is_done():
            return 0

        return len(new_state.swap_gates_inserted) * self._penalty_per_swap

    def _set_reward_range(self) -> None:
        """Set the reward range."""
        l_bound = -float("inf")
        if self._illegal_action_penalty >= 0 and self._penalty_per_swap >= 0:
            l_bound = 0

        u_bound = float("inf")
        if self._illegal_action_penalty <= 0 and self._penalty_per_swap <= 0:
            u_bound = 0

        self._reward_range = (l_bound, u_bound)
