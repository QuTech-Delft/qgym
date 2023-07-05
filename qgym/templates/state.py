"""Generic abstract base class for States of RL environments.

All states should inherit from ``State``.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from gymnasium.spaces import Space
from numpy.random import Generator, default_rng

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class State(Generic[ObservationT, ActionT]):
    """RL State containing the current state of the problem.

    Each subclass should set at least the following attributes:

    :ivar steps_done: Number of steps done since the last reset.
    """

    steps_done: int
    _rng: Optional[Generator] = None

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, **_kwargs: Any
    ) -> State[ObservationT, ActionT]:
        """Reset the state.

        :returns: Self"""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        self._rng = default_rng(seed)
        return [seed]

    @property
    def rng(self) -> Generator:
        """Return the random number generator of this environment. If none is set yet,
        this will generate a new one using ``numpy.random.default_rng``.

        :returns: Random number generator used by this ``Environment``.
        """
        if self._rng is None:
            self._rng = default_rng()
        return self._rng

    @rng.setter
    def rng(self, rng: Generator) -> None:
        self._rng = rng

    @abstractmethod
    def update_state(self, action: ActionT) -> State[ObservationT, ActionT]:
        """Update the state of this ``Environment`` using the given action.

        :param action: Action to be executed.
        """
        raise NotImplementedError

    @abstractmethod
    def obtain_observation(self) -> ObservationT:
        """:return: Observation based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        raise NotImplementedError

    @abstractmethod
    def obtain_info(self) -> Dict[Any, Any]:
        """:return: Optional debugging info for the current state."""
        raise NotImplementedError

    @abstractmethod
    def create_observation_space(self) -> Space[Any]:
        """Create the corresponding observation space."""
        raise NotImplementedError

    def __repr__(self) -> str:
        text = f"{self.__class__.__name__}:\n"
        if hasattr(self, "__slots__"):
            for attribute_name in self.__slots__:
                text += f"{attribute_name}: {repr(getattr(self, attribute_name))}\n"
        if hasattr(self, "__dir__"):
            for attribute_name, attribute_value in self.__dict__.items():
                text += f"{attribute_name}: {repr(attribute_value)}\n"
        return text
