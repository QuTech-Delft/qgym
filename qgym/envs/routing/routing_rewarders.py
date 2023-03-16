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


class BasicRewarder:
    """RL Rewarder, for computing rewards on a state."""

    _reward_range: Tuple[float, float]

    @abstractmethod
    def compute_reward(self, *, old_state: Any, action: Any, new_state: Any) -> float:
        """Compute a reward, based on the old state, new state, and the given action.

        :param old_state: State of the ``Environment`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``Environment``.
        :return reward: The reward for this action.
        """
        raise NotImplementedError
        #TODO: implement compute reward functionality

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Reward range of the rewarder. I.e., range that rewards can lie in."""
        return self._reward_range