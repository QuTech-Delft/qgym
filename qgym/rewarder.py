"""Generic abstract base class for a RL rewarder. All rewarders should inherit from
``Rewarder``.
"""
from abc import abstractmethod
from typing import Any, Tuple


class Rewarder:
    """RL Rewarder, for computing rewards on a state."""

    _reward_range: Tuple[float, float]

    @abstractmethod
    def compute_reward(self, *args: Any, **kwargs: Any) -> float:
        """Compute a reward, based on the given arguments.

        :param args: Arguments for computing the reward.
        :param kwargs: Keyword-arguments for computing the reward.
        """
        raise NotImplementedError

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Reward range of the rewarder. I.e., range that rewards can lie in."""
        return self._reward_range
