"""A dictionary of spaces, i.e. a dictionary with fixed string keys and space( sample)s
as values."""

import typing
from typing import Generator, Optional

import gym.spaces


class Dict(gym.spaces.Dict):
    """Dictionary of other action/observation spaces for use in RL environments."""

    def __init__(
        self,
        spaces: Optional[typing.Dict[str, gym.Space]] = None,
        rng: Optional[Generator] = None,
        **spaces_kwargs: gym.Space
    ):
        """Initialize a dictionary space, with string valued keys and Spaces as values.

        :param spaces: Dictionary containing string valued keys and spaces that are to
            form this Dictionary space.
        :param rng: Random number generator to be used in this space. If `None` a new
            one will be constructed.
        :param spaces_kwargs: Spaces that are to form this Dictionary space."""

        super(Dict, self).__init__(spaces, **spaces_kwargs)
        for space in self.spaces.values():
            space._np_random = (
                rng  # this overrides the default behaviour of the gym space
            )
