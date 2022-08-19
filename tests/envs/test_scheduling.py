import numpy as np
from stable_baselines3.common.env_checker import check_env

import qgym.spaces
from qgym.custom_types import Gate
from qgym.envs import Scheduling
from qgym.utils import GateEncoder

mp = {
    "qubit_number": 10,
    "gates": {
        "prep": 1,
        "x": 2,
        "y": 2,
        "z": 2,
        "h": 2,
        "cnot": 4,
        "swap": 3,
        "measure": 10,
    },
    "machine_restrictions": {
        "same_start": {"measure"},
        "not_in_same_cycle": {"x": ["y", "z"], "y": ["x", "z"], "z": ["x", "y"]},
    },
}


def naive_schedule_algorithm(scheduling_env, circuit=None):
    obs = scheduling_env.reset(circuit=circuit)
    action = np.array([0, 0])
    done = False

    while not done:
        while obs["legal_actions"].any():
            action[0] = obs["legal_actions"].argmax()
            obs, _, done, _ = scheduling_env.step(action)

        action[1] = 1
        obs, _, done, _ = scheduling_env.step(action)
        action[1] = 0
    return scheduling_env._state["schedule"]


def test_state() -> None:
    env = Scheduling(mp)

    assert isinstance(env._state["n_qubits"], int)
    assert env._state["n_qubits"] == mp["qubit_number"]

    assert isinstance(env._gate_encoder, GateEncoder)
    gate_encoder = env._gate_encoder
    assert hasattr(gate_encoder, "encoding_dct")
    assert hasattr(gate_encoder, "decoding_dct")
    assert hasattr(gate_encoder, "longest_name")
    assert hasattr(gate_encoder, "n_gates")
    assert gate_encoder.n_gates == len(mp["gates"])
    assert len(gate_encoder.encoding_dct) == len(mp["gates"])
    assert len(gate_encoder.encoding_dct) == len(mp["gates"])


def test_observation_space() -> None:
    env = Scheduling(mp)

    assert isinstance(env.observation_space, qgym.spaces.Dict)

    observation_space = [
        ("legal_actions", qgym.spaces.MultiBinary),
        ("gate_names", qgym.spaces.MultiDiscrete),
        ("acts_on", qgym.spaces.MultiDiscrete),
        ("dependencies", qgym.spaces.MultiDiscrete),
    ]

    for name, space_type in observation_space:
        space = env.observation_space[name]
        assert isinstance(space, space_type)


def test_action_space() -> None:
    env = Scheduling(mp, max_gates=100)

    assert isinstance(env.action_space, qgym.spaces.MultiDiscrete)

    assert [0, False] in env.action_space
    assert [0, 0] in env.action_space
    assert [0, 1] in env.action_space
    assert [0, True] in env.action_space
    assert [99, 0] in env.action_space
    assert [99, 1] in env.action_space

    assert [-1, 0] not in env.action_space
    assert [0, -1] not in env.action_space

    assert [100, 0] not in env.action_space
    assert [0, 2] not in env.action_space

    for _ in range(100):
        sample = env.action_space.sample()
        assert sample[0] in range(100)
        assert sample[1] in range(2)


def test_scheduled_after() -> None:
    env = Scheduling(mp, dependency_depth=2)
    circuit = [Gate("cnot", 1, 2), Gate("x", 2, 2), Gate("cnot", 1, 3)]
    obs = env.reset(circuit=circuit)

    expected_scheduled_after = np.zeros(400, dtype=int)
    expected_scheduled_after[0] = 1
    expected_scheduled_after[200] = 2

    assert (obs["dependencies"] == expected_scheduled_after).all()


def test_same_gates_commute() -> None:
    env = Scheduling(mp)
    circuit = [Gate("cnot", 1, 2), Gate("cnot", 1, 2)]
    obs = env.reset(circuit=circuit)

    expected_scheduled_after = np.zeros(200, dtype=int)

    assert (obs["dependencies"] == expected_scheduled_after).all()


def test_legal_actions() -> None:
    env = Scheduling(mp)
    circuit = [Gate("cnot", 1, 2), Gate("x", 2, 2), Gate("cnot", 1, 3)]
    obs = env.reset(circuit=circuit)

    expected_legal_actions = np.zeros(200, dtype=bool)
    expected_legal_actions[1] = True
    expected_legal_actions[2] = True

    assert (obs["legal_actions"] == expected_legal_actions).all()


def test_validity() -> None:
    env = Scheduling(mp)
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True


def test_full_cnot_circuit() -> None:
    env = Scheduling(mp)
    circuit = [
        Gate("cnot", 0, 1),
        Gate("cnot", 2, 3),
        Gate("cnot", 4, 5),
        Gate("cnot", 6, 7),
        Gate("cnot", 8, 9),
    ]
    schedule = naive_schedule_algorithm(env, circuit=circuit)
    assert (schedule == np.zeros(5)).all()


def test_same_start_machine_restriction() -> None:
    env = Scheduling(mp)
    circuit = [Gate("measure", 1, 1), Gate("y", 1, 1), Gate("measure", 0, 0)]
    schedule = naive_schedule_algorithm(env, circuit=circuit)
    assert (schedule == np.array([10, 0, 0])).all()


def test_not_in_same_cycle_machine_restriction() -> None:
    env = Scheduling(mp)
    circuit = [Gate("x", 0, 0), Gate("y", 1, 1), Gate("y", 3, 3), Gate("z", 2, 2)]
    schedule = naive_schedule_algorithm(env, circuit=circuit)
    assert (schedule == np.array([0, 2, 2, 4])).all()
