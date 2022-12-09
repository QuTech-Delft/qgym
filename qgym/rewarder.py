"""Generic abstract base class for a RL rewarder. All rewarders should inherit from
``Rewarder``.
"""
from abc import abstractmethod
from typing import Any, Dict, Tuple


class Rewarder:
    """RL Rewarder, for computing rewards on a state."""

    _reward_range: Tuple[float, float]

    @abstractmethod
    def compute_reward(
        self,
        *,
        old_state: Dict[str, Any],
        action: Any,
        new_state: Dict[str, Any],
    ) -> float:
        """Compute a reward, based on the old state, new state, and the given action.

        :param old_state: State of the ``Environment`` before the current action.
        :param action: Action that has just been taken.
        :param new_state: Updated state of the ``Environment``.
        :return reward: The reward for this action.
        """
        raise NotImplementedError

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Reward range of the rewarder. I.e., range that rewards can lie in."""
        return self._reward_range
