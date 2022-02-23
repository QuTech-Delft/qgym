from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from numpy.random import Generator, default_rng

ValueT = TypeVar("ValueT", covariant=True)


class Space(Generic[ValueT]):
    """
    Action or observation space of a RL environment.
    """
    def __init__(self, rng: Optional[Generator] = None):
        self._rng = rng

    @abstractmethod
    def sample(self) -> ValueT:
        """
        Sample a random value from this space.

        :return: Random value from this space.
        """
        raise NotImplementedError

    @property
    def rng(self) -> Generator:
        """
        The random number generator of this space. If none is set yet, this will generate a new one, with a
        random seed.
        """
        if self._rng is None:
            self._rng = default_rng()
        return self._rng

    @rng.setter
    def rng(self, rng: Generator) -> None:
        self._rng = rng
