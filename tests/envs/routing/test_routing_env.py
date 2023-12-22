from __future__ import annotations

import pytest
from stable_baselines3.common.env_checker import check_env

from qgym.envs.routing.routing import Routing


@pytest.mark.parametrize(
    "kwargs",
    [
        {"connection_grid_size": (2, 2)},
        {"connection_grid_size": (2, 2), "observe_legal_surpasses": False},
        {"connection_grid_size": (2, 2), "observe_connection_graph": False},
        {
            "connection_grid_size": (2, 2),
            "observe_legal_surpasses": False,
            "observe_connection_graph": False,
        },
    ],
)
def test_validity(kwargs: dict[str, tuple[int, int] | bool]) -> None:
    env = Routing(**kwargs)  # type: ignore[arg-type]
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
