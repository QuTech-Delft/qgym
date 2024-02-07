"""This package contains general spaces that are to be used as action or observation
space in custom RL Environments. All spaces inherit from spaces of the ``gymnasium``
package.
"""

from qgym.spaces.box import Box
from qgym.spaces.dict import Dict
from qgym.spaces.discrete import Discrete
from qgym.spaces.multi_binary import MultiBinary
from qgym.spaces.multi_discrete import MultiDiscrete

__all__ = ["Box", "Dict", "Discrete", "MultiBinary", "MultiDiscrete"]
