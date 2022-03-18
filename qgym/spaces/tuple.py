from typing import Any, Iterable, Optional, Tuple

from numpy.random import Generator

from qgym import Space


class TupleSpace(Space[Tuple[Any]]):
    def __init__(self, spaces: Iterable[Space[Any]], rng: Optional[Generator] = None):
        """
        Initialize a tuple space, i.e. a tuple of other spaces.

        :param spaces: Other spaces that form this tuple space in order.
        :param rng: Random number generator to be used in this space. If `None` a new one will be constructed.
        """
        super().__init__(rng)
        self._spaces = tuple(spaces)
        for space in self._spaces:
            if not isinstance(space, Space):
                raise ValueError("All provided spaces should be a subclass of `Space`.")

    def sample(self) -> Tuple[Any]:
        """
        Sample a random value from this space.

        :return: Random value from this space.
        """
        return tuple(space.sample() for space in self._spaces)

    def __contains__(self, values: Any) -> bool:
        if isinstance(values, Iterable):
            values = tuple(values)
            if len(values) == len(self._spaces):
                return all(value in space for value, space in zip(values, self._spaces))
        return False

    def __str__(self) -> str:
        return f"TupleSpace({', '.join(str(space) for space in self._spaces)})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TupleSpace) and self._spaces == other._spaces
