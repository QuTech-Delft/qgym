"""Specific environments of this RL Gym in the Quantum domain. This package
contains the :class:`InitialMapping`, :class:`Routing` and :class:`Scheduling`
environments, which model their respective OpenQL passes.
"""

from qgym.envs.initial_mapping.initial_mapping import InitialMapping
from qgym.envs.routing import Routing
from qgym.envs.scheduling.scheduling import Scheduling

__all__ = ["InitialMapping", "Routing", "Scheduling"]
