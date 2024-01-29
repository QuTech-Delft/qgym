"""Generic abstract base class for RL environments.

All environments should inherit from ``Environment``.
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Any, Mapping

import gymnasium
import numpy as np
from gymnasium import Space
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from qgym.templates.rewarder import Rewarder
from qgym.templates.state import ActionT, ObservationT, State
from qgym.templates.visualiser import Visualiser


class Environment(gymnasium.Env[ObservationT, ActionT]):
    """RL Environment containing the current state of the problem.

    Each subclass should set at least the following attributes:
    """

    # --- These attributes should be set in any subclass ---
    action_space: Space[Any]
    """The action space of this environment."""
    observation_space: Space[Any]
    """The observation space of this environment."""
    metadata: dict[str, Any]
    """Additional metadata of this environment."""
    _state: State[ObservationT, ActionT]
    """The state space of this environment."""
    _rewarder: Rewarder
    """The rewarder of this environment."""
    _visualiser: Visualiser | None = None
    """The visualiser of this environment."""

    # --- Other attributes ---
    _rng: Generator | None = None

    def step(
        self, action: ActionT
    ) -> tuple[ObservationT, float, bool, bool, dict[Any, Any]]:
        """Update the state based on the input action.

        Return observation, reward, done-indicator, terminated-indicator and debugging
        info based on the updated state.

        Args:
            action: Action to be performed.

        Returns:
            A tuple containing five entries

            1. The updated state;
            2. Reward for the given action;
            3. Boolean value stating whether the new state is a final state (i.e., if
               we are done);
            4. Boolean value stating whether the episode is truncated. Currently always
               returns ``False``.
            5. Additional (debugging) information.
        """
        old_state = deepcopy(self._state)
        self._state.update_state(action)
        if self._visualiser is not None:
            self._visualiser.step(self._state)

        return (
            self._state.obtain_observation(),
            self._compute_reward(old_state, action),
            self._state.is_done(),
            False,
            self._state.obtain_info(),
        )

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[ObservationT, dict[str, Any]]:
        """Reset the environment and load a new random initial state.

        To be used after an episode is finished. Optionally, one can provide additional
        options to configure the reset.

        Args:
            seed: Seed for the random number generator, should only be provided
                (optionally) on the first reset call, i.e., before any learning is done.
            options: Dictionary containing keyword-argument pairs to configure the
                reset.

        Returns:
            Initial observation and a dictionary containing debugging information.
        """
        super().reset(seed=seed)
        options = {} if options is None else options
        self._state.reset(seed=seed, **options)
        self.render()
        return self._state.obtain_observation(), self._state.obtain_info()

    def render(self) -> None | NDArray[np.int_]:  # type: ignore[override]
        """Render the current state using pygame.

        Returns:
            Result of rendering.
        """
        if self._visualiser is not None:
            return self._visualiser.render(self._state)
        return None

    def close(self) -> None:
        """Close the screen used for rendering."""
        if self._visualiser is not None:
            self._visualiser.close()
        self._visualiser = None

    @property
    def rewarder(self) -> Rewarder:
        """Return the rewarder that is set for this environment.

        Used to compute rewards after each step.
        """
        return self._rewarder

    @rewarder.setter
    def rewarder(self, rewarder: Rewarder) -> None:
        self._rewarder = rewarder
        self.reward_range = rewarder.reward_range

    @property
    def rng(self) -> Generator:
        """Return the random number generator of this environment.

        If none is set yet, this will generate a new one using
        ``numpy.random.default_rng``.
        """
        if self._rng is None:
            self._rng = default_rng()
        return self._rng

    @rng.setter
    def rng(self, rng: Generator) -> None:
        self._rng = rng

    def __del__(self) -> None:
        if hasattr(self, "_visualiser"):
            self.close()

    def _compute_reward(
        self,
        old_state: State[ObservationT, ActionT],
        action: ActionT,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """Ask the ``Rewarder`` to compute a reward, based on the given old state, the
        given action and the updated state.

        Args:
            old_state: The state of the ``Environment`` before the action was taken.
            action: Action that was taken.
            args: Optional arguments for the ``Rewarder``.
            kwargs: Optional keyword-arguments for the ``Rewarder``.
        """
        return self._rewarder.compute_reward(
            *args, old_state=old_state, action=action, new_state=self._state, **kwargs
        )
