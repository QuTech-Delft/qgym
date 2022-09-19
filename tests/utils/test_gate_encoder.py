import pytest
from qgym.utils import GateEncoder
from qgym.custom_types import Gate


@pytest.fixture
def empty_encoder():
    return GateEncoder()

@pytest.fixture
def trained_encoder(empty_encoder):
    return empty_encoder.learn_gates(["x", "y", "z", "cnot", "h"])
    

@pytest.mark.parametrize(
    "gates",
    [["x", "y", "z", "cnot", "h"],
     ("x", "y", "z", "cnot", "h"),
     {"x":1, "y":1, "z":1, "cnot":1, "h":1},
     ],
)
def test_learn_gates(empty_encoder, gates):
    
    assert empty_encoder.encoding_dct == None
    assert empty_encoder.decoding_dct == None
    assert empty_encoder.longest_name == None
    assert empty_encoder.n_gates == None
    
    assert type(empty_encoder.learn_gates(gates)) == GateEncoder
    
    assert empty_encoder.encoding_dct == {"x":1, "y":2, "z":3, "cnot":4, "h": 5}
    assert empty_encoder.decoding_dct == {1:"x", 2:"y", 3:"z", 4:"cnot", 5: "h"}
    assert empty_encoder.longest_name == 4
    assert empty_encoder.n_gates == 5
    

def test_duplicate_gates_warning(empty_encoder):
    with pytest.warns(UserWarning):
        empty_encoder.learn_gates(["x", "y", "z", "cnot", "h", "h", "z"])
    
    assert empty_encoder.encoding_dct == {"x":1, "y":2, "z":3, "cnot":4, "h": 5}
    assert empty_encoder.decoding_dct == {1:"x", 2:"y", 3:"z", 4:"cnot", 5: "h"}
    assert empty_encoder.longest_name == 4
    assert empty_encoder.n_gates == 5

@pytest.mark.parametrize(
    "gates, encoded_gates",
    [
     ("x", 1),
     ({"cnot":1, "z":2}, {4:1, 3:2}),
     ([Gate("y",1,1), Gate("h",1,2)], [Gate(2,1,1), Gate(5,1,2)]),
     (["x", "cnot", "y"], [1, 4, 2])
     ],
)
def test_encode_gates(trained_encoder, gates, encoded_gates):
    assert trained_encoder.encode_gates(gates) == encoded_gates


@pytest.mark.parametrize(
    "gates, encoded_gates",
    [
     ("x", 1),
     ({"cnot":1, "z":2}, {4:1, 3:2}),
     ([Gate("y",1,1), Gate("h",1,2)], [Gate(2,1,1), Gate(5,1,2)]),
     (["x", "cnot", "y"], [1, 4, 2])
     ],
)
def test_decode_gates(trained_encoder, gates, encoded_gates):
    assert trained_encoder.decode_gates(encoded_gates) == gates
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    