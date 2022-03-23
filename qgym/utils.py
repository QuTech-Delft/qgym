"""
Generic utils for the Quantum RL Gym
"""
from typing import Any

import numpy as np
from numpy.typing import NDArray


def check_adjacency_matrix(adjacency_matrix: NDArray[Any]) -> None:
    """
    :param adjacency_matrix: Matrix to check.
    :raise ValueError: In case the provided input is not a valid matrix.
    """
    if (
        not adjacency_matrix.ndim == 2
        and adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    ):
        raise ValueError("The provided value should be a square 2-D adjacency matrix.")
