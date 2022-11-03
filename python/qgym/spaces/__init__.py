"""This package contains general spaces that are to be used as action or observation
space in custom RL Environments. All spaces inherit from spaces of the OpenAI
``gym.spaces`` package.
"""
from qgym.spaces.box import Box as Box
from qgym.spaces.dict import Dict as Dict
from qgym.spaces.multi_binary import MultiBinary as MultiBinary
from qgym.spaces.multi_discrete import MultiDiscrete as MultiDiscrete
