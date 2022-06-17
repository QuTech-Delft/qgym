import pytest
from stable_baselines3.common.env_checker import check_env

import qgym.spaces
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
    "machine_restrictions": set(),
    "commutation_rules": set(),
}


def test_state() -> None:
    env = Scheduling(mp)

    assert isinstance(env._state["n_qubits"], int)
    assert env._state["n_qubits"] == mp["qubit_number"]

    assert isinstance(env._state["gate_encoder"], GateEncoder)
    gate_encoder = env._state["gate_encoder"]
    assert hasattr(gate_encoder, "encoding_dct")
    assert hasattr(gate_encoder, "decoding_dct")
    assert hasattr(gate_encoder, "longest_name")
    assert hasattr(gate_encoder, "n_gates")
    assert gate_encoder.n_gates == len(mp["gates"])
    assert len(gate_encoder.encoding_dct) == len(mp["gates"])
    assert len(gate_encoder.encoding_dct) == len(mp["gates"])

    assert isinstance(env._state["encoded_gates"], dict)
    encoded_gates = env._state["encoded_gates"]
    decoded_gates = gate_encoder.decode_gates(encoded_gates)
    assert decoded_gates == mp["gates"]


def test_observation_space() -> None:
    env = Scheduling(mp)

    assert isinstance(env.observation_space, qgym.spaces.Dict)

    observation_space = [
        ("legal_actions", qgym.spaces.MultiBinary),
        ("gate_names", qgym.spaces.MultiDiscrete),
        ("acts_on", qgym.spaces.MultiDiscrete),
        ("scheduled_after", qgym.spaces.MultiDiscrete),
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


def test_validity() -> None:
    env = Scheduling(mp)
    check_env(env, warn=True)  # todo: maybe switch this to the gym env checker
    assert True
