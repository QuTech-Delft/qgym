"""
General spaces to be used as action or observation space in RL Environments.
"""

from qgym.spaces.adaptive_multi_discrete import (
    AdaptiveMultiDiscrete as AdaptiveMultiDiscrete,
)
from qgym.spaces.discrete import Discrete as Discrete
from qgym.spaces.injective_partial_map import InjectivePartialMap as InjectivePartialMap
from qgym.spaces.multi_discrete import MultiDiscrete as MultiDiscrete
