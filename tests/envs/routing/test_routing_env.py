from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from qgym.envs.routing.routing import Routing


class TestEnvironment:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"connection_graph": (2, 2)},
            {"connection_graph": (2, 2), "observe_legal_surpasses": False},
            {"connection_graph": (2, 2), "observe_connection_graph": False},
            {
                "connection_graph": (2, 2),
                "observe_legal_surpasses": False,
                "observe_connection_graph": False,
            },
        ],
    )
    def test_validity(self, kwargs: dict[str, tuple[int, int] | bool]) -> None:
        env = Routing(**kwargs)  # type: ignore[arg-type]
        check_env(env, warn=True)  # todo: maybe switch this to the gym env checker

    def test_step(self) -> None:
        env = Routing(connection_graph=[2, 2])  # type: ignore[arg-type]
        obs = env.step(0)[0]
        np.testing.assert_array_equal(obs["mapping"], [2, 1, 0, 3])

    @pytest.mark.parametrize(
        ("actions", "mapping"),
        [
            ([0, 1], [2, 0, 1]),  # swap(q0, q2), swap(q1, q2)
            ([1, 0], [1, 2, 0]),  # swap(q1, q2), swap(q0, q2)
        ]
    )
    def test_multiple_steps(self, actions,
                            mapping) -> None:
        env = Routing(connection_graph=[1, 3])

        for action in actions:
            obs = env.step(action)[0]

        # noinspection PyUnboundLocalVariable
        np.testing.assert_array_equal(obs["mapping"], mapping)
