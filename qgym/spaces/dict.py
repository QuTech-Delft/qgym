"""
A dictionary of spaces, i.e., a dictionary with fixed strings as keys and spaces as
values.
"""

import typing
from typing import List, Optional

import gym.spaces
from numpy.random import Generator, default_rng


class Dict(gym.spaces.Dict):
    """
    Dictionary of other action/observation spaces for use in RL environments.
    """

    def __init__(
        self,
        spaces: Optional[typing.Dict[str, gym.Space]] = None,
        rng: Optional[Generator] = None,
        **spaces_kwargs: gym.Space
    ) -> None:
        """
        Initialize a dictionary space, with string valued keys and Spaces as values.

        :param spaces: Dictionary containing string valued keys and spaces that are to
            form this Dictionary space.
        :param rng: Random number generator to be used in this space, if `None` a new
            one will be constructed.
        :param spaces_kwargs: Spaces that are to form this Dictionary space.
        """

        super(Dict, self).__init__(spaces, **spaces_kwargs)
        for space in self.spaces.values():
            # override the default behaviour of the gym space
            space._np_random = rng

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Seed the rng of this space.

        :param seed: Seed for the rng
        :return: The used seeds.
        """
        for space in self.spaces.values():
            space._np_random = default_rng(seed)
        return [seed]
