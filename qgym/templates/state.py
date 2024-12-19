"""Generic abstract base class for States of RL environments.

All states should inherit from ``State``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from gymnasium.spaces import Space
from numpy.random import Generator, default_rng

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class State(Generic[ObservationT, ActionT]):
    """RL State containing the current state of the problem."""

    steps_done: int
    """Number of steps done since the last reset."""
    _rng: Generator | None = None

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, **_kwargs: Any
    ) -> State[ObservationT, ActionT]:
        """Reset the state.

        Returns:
            Self.
        """
        raise NotImplementedError

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        Args:
            seed: Seed for the rng. Defaults to ``None``

        Returns:
            The used seeds.
        """
        self._rng = default_rng(seed)
        return [seed]

    @property
    def rng(self) -> Generator:
        """Return the random number generator of this environment. If none is set yet,
        this will generate a new one using ``numpy.random.default_rng``.

        Returns:
            Random number generator used by this ``Environment``.
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

        Args:
            action: Action to be executed.

        Returns:
            Self.
        """
        raise NotImplementedError

    @abstractmethod
    def obtain_observation(self) -> ObservationT:
        """Observation based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        """Boolean value stating whether we are in a final state."""
        raise NotImplementedError

    def is_truncated(self) -> bool:
        """Boolean value stating whether the episode is truncated."""
        return False

    @abstractmethod
    def obtain_info(self) -> dict[Any, Any]:
        """Optional debugging info for the current state."""
        raise NotImplementedError

    @abstractmethod
    def create_observation_space(self) -> Space[Any]:
        """Create the corresponding observation space."""
        raise NotImplementedError

    def __repr__(self) -> str:
        text = f"{self.__class__.__name__}:\n"
        if hasattr(self, "__slots__"):
            for attribute_name in self.__slots__:
                text += f"{attribute_name}: {getattr(self, attribute_name)!r}\n"
        if hasattr(self, "__dir__"):
            for attribute_name, attribute_value in self.__dict__.items():
                text += f"{attribute_name}: {attribute_value!r}\n"
        return text
