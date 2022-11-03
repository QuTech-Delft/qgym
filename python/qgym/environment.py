"""Generic abstract base class for RL environments. All environments should inherit
from ``Environment``.
"""
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import gym
from gym import Space
from numpy.random import Generator, default_rng
from qgym.rewarder import Rewarder

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
SelfT = TypeVar("SelfT")


class Environment(Generic[ObservationT, ActionT], gym.Env):
    """RL Environment containing the current state of the problem.

    Each subclass should set at least the following attributes:

    :ivar action_space: The action space of this environment.
    :ivar observation_space: The observation space of this environment.
    :ivar metadat: Additional metadata of this environment.
    :ivar _state: The state space of this environment.
    :ivar _rewarder: The rewarder of this environment.
    """

    # --- These attributes should be set in any subclass ---
    action_space: Space
    observation_space: Space
    metadata: Dict[str, Any]
    _state: Dict[str, Any]
    _rewarder: Rewarder

    # --- Other attributes ---
    _rng: Optional[Generator] = None

    def step(
        self, action: ActionT, *, return_info: bool = True
    ) -> Union[
        Tuple[ObservationT, float, bool],
        Tuple[ObservationT, float, bool, Dict[Any, Any]],
    ]:
        """Update the state based on the input action. Return observation, reward,
        done-indicator and (optional) debugging info based on the updated state.

        :param action: Action to be performed.
        :param return_info: Whether to receive debugging info. Defaults to ``False``.
        :return: A tuple containing three/four entries

            1. The updated state;
            2. Reward for the given action;
            3. Boolean value stating whether the new state is a final state (i.e., if
               we are done);
            4. Optional additional (debugging) information. This entry is only given if
               `return_info` is ``True``.
        """
        old_state = deepcopy(self._state)
        self._update_state(action)
        if return_info:
            return (
                self._obtain_observation(),
                self._compute_reward(old_state, action),
                self._is_done(),
                self._obtain_info(),
            )
        return (
            self._obtain_observation(),
            self._compute_reward(old_state, action),
            self._is_done(),
        )

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, return_info: bool = False, **_kwargs: Any
    ) -> Union[ObservationT, Tuple[ObservationT, Dict[Any, Any]]]:
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
        if seed is not None:
            self.seed(seed)

        if return_info:
            return self._obtain_observation(), self._obtain_info()
        return self._obtain_observation()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the rng of this space, using ``numpy.random.default_rng``.

        :param seed: Seed for the rng. Defaults to ``None``
        :return: The used seeds.
        """
        self._rng = default_rng(seed)
        return [seed]

    @abstractmethod
    def render(self, mode: str = "human") -> None:
        """Render the current state.

        :param mode: The mode to render with. Defaults to 'human'.
        """
        raise NotImplementedError

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
        self, old_state: Dict[str, Any], action: ActionT, *args: Any, **kwargs: Any
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

    @abstractmethod
    def _update_state(self, action: ActionT) -> None:
        """Update the state of this ``Environment`` using the given action.

        :param action: Action to be executed.
        """
        raise NotImplementedError

    @abstractmethod
    def _obtain_observation(self) -> ObservationT:
        """:return: Observation based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def _is_done(self) -> bool:
        """:return: Boolean value stating whether we are in a final state."""
        raise NotImplementedError

    @abstractmethod
    def _obtain_info(self) -> Dict[Any, Any]:
        """:return: Optional debugging info for the current state."""
        raise NotImplementedError
