from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

from numpy.random import Generator, default_rng

from qgym.rewarder import Rewarder
from qgym.space import Space

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
SelfT = TypeVar("SelfT")


class Environment(Generic[ObservationT, ActionT]):
    """
    RL Environment containing the current state of the problem.
    """

    # --- These properties should be set in any subclass ---
    _action_space: Space[ActionT]
    _observation_space: Space[ObservationT]
    _state: Dict[Any, Any]
    _rewarder: Rewarder
    _metadata: Dict[Any, Any]

    # --- Other properties ---
    _rng: Optional[Generator] = None

    def step(
        self, action: ActionT, *, return_info: bool = False
    ) -> Union[
        Tuple[ObservationT, float, bool],
        Tuple[ObservationT, float, bool, Dict[Any, Any]],
    ]:
        """
        Update the state based on the input action. Return observation, reward, done-indicator and (optional) debugging
        info based on the updated state.

        :param action: Valid action to take.
        :param return_info: Whether to receive debugging info.
        :return: A tuple containing three/four entries: 1) The updated state; 2) Reward of the new state; 3) Boolean
            value stating whether the new state is a final state (i.e. if we are done); 4) Optional Additional
            (debugging) information.
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
        self, *, seed: Optional[int] = None, return_info: bool = False, **kwargs: Any
    ) -> Union[ObservationT, Tuple[ObservationT, Dict[Any, Any]]]:
        """
        Reset the environment and load a new random initial state. To be used after an episode is finished. Optionally,
        one can provide additional options to configure the reset.

        :param seed: Seed for the random number generator, should only be provided (optionally) on the first reset call,
            i.e. before any learning is done.
        :param return_info: Whether to receive debugging info.
        :param kwargs: Additional options to configure the reset. To be defined for a specific environment
        :return: Initial observation and optional debugging info.
        """
        if seed is not None:
            self._rng = default_rng(seed)

        if return_info:
            return self._obtain_observation(), self._obtain_info()
        return self._obtain_observation()

    @abstractmethod
    def render(self) -> None:
        """
        Render the current state.
        """
        pass

    def close(self) -> None:
        """
        This method is called on this object being removed. Should take care of loose ends. To be implemented in
        specific subclasses.
        """
        pass

    @property
    def action_space(self) -> Space[ActionT]:
        """
        Description of the current action space, i.e. the set of allowed actions (potentially depending on the current
        state).
        """
        return self._action_space

    @property
    def metadata(self) -> Dict[Any, Any]:
        """
        Metadata of this environment.
        """
        return self._metadata

    @property
    def observation_space(self) -> Space[ObservationT]:
        """
        Description of the observation space, i.e. how observations can look like.
        """
        return self.observation_space

    @property
    def rewarder(self) -> Rewarder:
        """
        The rewarder that is currently set for this environment. Is used to compute rewards after each step.
        """
        return self._rewarder

    @rewarder.setter
    def rewarder(self, rewarder: Rewarder) -> None:
        self._rewarder = rewarder

    @property
    def reward_range(self) -> Tuple[float, float]:
        """
        Reward range of the rewarder. I.e. range that rewards can lie in.
        """
        return self._rewarder.reward_range

    @property
    def rng(self) -> Generator:
        """
        The random number generator of this environment. If none is set yet, this will generate a new one, with a
        random seed.
        """
        if self._rng is None:
            self._rng = default_rng()
        return self._rng

    @rng.setter
    def rng(self, rng: Generator) -> None:
        self._rng = rng

    def __del__(self) -> None:
        self.close()

    @abstractmethod
    def _compute_reward(
        self, old_state: Dict[Any, Any], action: ActionT, *args: Any, **kwargs: Any
    ) -> float:
        """
        Asks the rewarder to compute a reward, based on the given arguments.

        :param args: Arguments for the Rewarder.
        :param kwargs: Keyword-arguments for the Rewarder.
        """
        return self._rewarder.compute_reward(
            *args, old_state=old_state, action=action, **kwargs
        )


    @abstractmethod
    def _update_state(self, action: ActionT) -> None:
        """
        Update the state of this environment using the given action.

        :param action: Action to be executed.
        """
        raise NotImplementedError

    @abstractmethod
    def _obtain_observation(self) -> ObservationT:
        """
        :return: Observation based on the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def _is_done(self) -> bool:
        """
        :return: Boolean value stating whether we are in a final state.
        """
        raise NotImplementedError

    @abstractmethod
    def _obtain_info(self) -> Dict[Any, Any]:
        """
        :return: Optional debugging info for the current state.
        """
        raise NotImplementedError
