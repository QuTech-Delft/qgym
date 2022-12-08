import pytest
from qgym.envs.scheduling import MachineProperties


@pytest.fixture
def empty_mp():
    return MachineProperties(1)


@pytest.fixture
def mp_with_gates(empty_mp):
    return empty_mp.add_gates({"x": 1, "y": 2})


@pytest.fixture
def diamond_mp():
    mp = MachineProperties(10)
    mp.add_gates(
        {"prep": 1, "x": 2, "y": 2, "z": 2, "h": 2, "cnot": 4, "swap": 3, "measure": 10}
    )
    mp.add_same_start(["measure"])
    mp.add_not_in_same_cycle([("x", "y"), ("x", "z"), ("y", "z")])
    return mp


@pytest.fixture
def diamond_mp_dict():
    return {
        "n_qubits": 10,
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


@pytest.mark.parametrize(
    "attribute",
    ["n_qubits", "gates", "same_start", "not_in_same_cycle", "n_gates"],
)
def test_attribute_errors(empty_mp, attribute):
    assert hasattr(empty_mp, attribute)
    with pytest.raises(AttributeError):
        setattr(empty_mp, attribute, None)


def test_add_gates(empty_mp):
    assert empty_mp.n_gates == 0

    empty_mp.add_gates({"x": 1, "y": 2})
    assert empty_mp.gates == {"x": 1, "y": 2}
    assert empty_mp.n_gates == 2

    empty_mp.add_gates({"Z": 3.0})
    assert empty_mp.gates == {"x": 1, "y": 2, "z": 3}
    assert empty_mp.n_gates == 3

    with pytest.warns(UserWarning) as record:
        empty_mp.add_gates({"x": 4})
        assert empty_mp.gates == {"x": 4, "y": 2, "z": 3}
        assert empty_mp.n_gates == 3

    assert len(record) == 1


@pytest.mark.parametrize(
    "known_gates,unknown_gates",
    [
        (["x", "y"], ["z"]),
        (("x", "y"), ("z",)),
        ({"x", "y"}, {"z"}),
    ],
)
def test_add_same_start(mp_with_gates, known_gates, unknown_gates):
    mp_with_gates.add_same_start(known_gates)
    assert mp_with_gates.same_start == {"x", "y"}
    with pytest.raises(ValueError):
        mp_with_gates.add_same_start(unknown_gates)


@pytest.mark.parametrize(
    "known_gates,unknown_gates",
    [
        ([("x", "y")], [("x", "z")]),
        ((("x", "y"),), (("x", "z"),)),
        ({("x", "y")}, {("x", "z")}),
    ],
)
def test_add_not_in_same_cycle(mp_with_gates, known_gates, unknown_gates):
    mp_with_gates.add_not_in_same_cycle(known_gates)
    assert mp_with_gates.not_in_same_cycle == {"x": ["y"], "y": ["x"]}
    with pytest.raises(ValueError):
        mp_with_gates.add_not_in_same_cycle(unknown_gates)


def test_diamond_mp(diamond_mp, diamond_mp_dict):
    assert diamond_mp.n_qubits == diamond_mp_dict["n_qubits"]
    assert diamond_mp.gates == diamond_mp_dict["gates"]
    assert (
        diamond_mp.same_start == diamond_mp_dict["machine_restrictions"]["same_start"]
    )
    assert (
        diamond_mp.not_in_same_cycle
        == diamond_mp_dict["machine_restrictions"]["not_in_same_cycle"]
    )


def test_from_mapping(diamond_mp_dict):
    mp = MachineProperties.from_mapping(diamond_mp_dict)
    assert mp.n_qubits == diamond_mp_dict["n_qubits"]
    assert mp.gates == diamond_mp_dict["gates"]
    assert mp.same_start == diamond_mp_dict["machine_restrictions"]["same_start"]
    assert (
        mp.not_in_same_cycle
        == diamond_mp_dict["machine_restrictions"]["not_in_same_cycle"]
    )

    diamond_mp_dict["machine_restrictions"].pop("same_start")
    error_msg = "'machine_restrictions' must have the keys 'same_start' and "
    error_msg += "'not_in_same_cycle'"
    with pytest.raises(ValueError, match=error_msg):
        MachineProperties.from_mapping(diamond_mp_dict)

    diamond_mp_dict.pop("machine_restrictions")
    error_msg = "'machine_properties' must have the keys 'n_qubits', 'gates' and "
    error_msg += "'machine_restrictions'"
    with pytest.raises(ValueError, match=error_msg):
        MachineProperties.from_mapping(diamond_mp_dict)
