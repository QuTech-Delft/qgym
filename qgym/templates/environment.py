"""Generic abstract base class for RL environments.

All environments should inherit from ``Environment``.
"""
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Tuple, Union

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

    :ivar action_space: The action space of this environment.
    :ivar observation_space: The observation space of this environment.
    :ivar metadat: Additional metadata of this environment.
    :ivar _state: The state space of this environment.
    :ivar _rewarder: The rewarder of this environment.
    :ivar _visualiser: The visualiser of this environment.
    """

    # --- These attributes should be set in any subclass ---
    action_space: Space[Any]
    observation_space: Space[Any]
    metadata: Dict[str, Any]
    _state: State[ObservationT, ActionT]
    _rewarder: Rewarder
    _visualiser: Optional[Visualiser]

    # --- Other attributes ---
    _rng: Optional[Generator] = None

    def step(
        self, action: ActionT
    ) -> Tuple[ObservationT, float, bool, bool, Dict[Any, Any]]:
        """Update the state based on the input action. Return observation, reward,
        done-indicator and (optional) debugging info based on the updated state.

        :param action: Action to be performed.
        :param return_info: Whether to receive debugging info. Defaults to ``False``.
        :return: A tuple containing three/four entries

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
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Tuple[ObservationT, Dict[str, Any]]:
        """Reset the environment and load a new random initial state. To be used after
        an episode is finished. Optionally, one can provide additional options to
        configure the reset.

        :param seed: Seed for the random number generator, should only be provided
            (optionally) on the first reset call, i.e., before any learning is done.
        :param return_info: Whether to receive debugging info. Defaults to ``False``.
        :param _kwargs: Additional keyword arguments to configure the reset. To be
            defined for a specific environment.
        :return: Initial observation and optionally also debugging info.
        """
        super().reset(seed=seed)
        if options is None:
            options = {}
        self._state.reset(seed=seed, **options)
        self.render()
        return self._state.obtain_observation(), self._state.obtain_info()

    def render(self) -> Union[None, NDArray[np.int_]]:  # type: ignore[override]
        """Render the current state using pygame.

        :return: Result of rendering.
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
        """Return the rewarder that is set for this environment. Used to compute rewards
        after each step.

        :returns: Rewarder of this ``Environment``.
        """
        return self._rewarder

    @rewarder.setter
    def rewarder(self, rewarder: Rewarder) -> None:
        self._rewarder = rewarder
        self.reward_range = rewarder.reward_range

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

    def __del__(self) -> None:
        self.close()

    def _compute_reward(
        self,
        old_state: State[ObservationT, ActionT],
        action: ActionT,
        *args: Any,
        **kwargs: Any
    ) -> float:
        """Ask the ``Rewarder`` to compute a reward, based on the given old state, the
        given action and the updated state.

        :param old_state: The state of the ``Environment`` before the action was taken.
        :param action: Action that was taken.
        :param args: Optional arguments for the ``Rewarder``.
        :param kwargs: Optional keyword-arguments for the ``Rewarder``.
        """
        return self._rewarder.compute_reward(
            *args, old_state=old_state, action=action, new_state=self._state, **kwargs
        )
