import pytest
from stable_baselines3.common.env_checker import check_env

from qgym.envs.routing.routing import Routing


@pytest.fixture(name="small_env")
def small_env_fixture() -> Routing:
    return Routing(connection_grid_size=(2, 2))


def test_validity(small_env: Routing) -> None:
    check_env(small_env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True
