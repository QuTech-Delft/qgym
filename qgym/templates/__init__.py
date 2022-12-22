"""This package contains templates used from buidling custom environments.

All ``Environment``s should inherit from ``Environment`` and should contain a rewarder,
state and visualiser which inherit from the base classes ``Rewarder``, ``State`` and
``Visualiser`` respectively.
"""

from qgym.templates.environment import Environment
from qgym.templates.rewarder import Rewarder
from qgym.templates.state import State
from qgym.templates.visualiser import Visualiser

__all__ = ["Environment", "Rewarder", "State", "Visualiser"]
