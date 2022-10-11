"""
Custom type definitions to ease type hinting.
"""

from collections import namedtuple

Gate = namedtuple("Gate", ["name", "q1", "q2"])
