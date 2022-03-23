from stable_baselines3.common.env_checker import check_env

from qgym.envs import InitialMapping


def test_validity() -> None:
    env = InitialMapping(
        connection_grid_size=(3, 3), interaction_graph_edge_probability=0.5
    )
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True
