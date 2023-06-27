"""Generic abstract base class for a RL rewarder. All rewarders should inherit from
``Rewarder``.
"""
from abc import abstractmethod
from typing import Any, Tuple


class Rewarder:
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

    @property
    def reward_range(self) -> Tuple[float, float]:
        """Reward range of the rewarder. I.e., range that rewards can lie in."""
        return self._reward_range

    def __eq__(self, other: Any) -> bool:
        """Checks whether an object 'other' is equal to self.

        This check is performed by checking of the self and other are of exactly the
        same type. Afterwards, all slots (if any) are equal and if all attributes are
        equal.

        returns: a boolean.
        """
        if type(self) is not type(other):
            return False

        if hasattr(self, "__slots__"):
            for attr in self.__slots__:
                if getattr(self, attr) != getattr(other, attr):
                    return False

        if hasattr(self, "__dict__") and self.__dict__ != other.__dict__:
            return False

        return True
