"""Specific benchmarks to test the performance of compilers. This package
contains the :class:`InitialMappingSolutionQuality` metrics.
"""

from qgym.benchmarks.metrics.initial_mapping_metrics import (
    DistanceRatioLoss,
    InitialMappingBenchmarker,
    InitialMappingMetric,
)

__all__ = ["DistanceRatioLoss", "InitialMappingBenchmarker", "InitialMappingMetric"]
