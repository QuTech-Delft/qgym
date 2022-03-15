"""
Generic utils for the Quantum RL Gym
"""
from typing import Any

import numpy as np
from numpy.typing import NDArray


def check_adjacency_matrix(adjacency_matrix: NDArray[Any]) -> bool:
    """
    :param adjacency_matrix: Matrix to check.
    :return: Whether this matrix could be a valid adjacency matrix.
    """
    if (
        not adjacency_matrix.ndim == 2
        and adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    ):
        raise ValueError("The provided value should be a square 2-D adjacency matrix.")
