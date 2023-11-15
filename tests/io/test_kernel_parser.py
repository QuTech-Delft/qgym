import pytest

from qgym.io.kernel_parser import qubit_to_int


@pytest.mark.parametrize("qubit,expected_result", [("q[0]", 0), (" \tq[2\r]\n", 2)])
def test_qubit_to_int(qubit: str, expected_result: int) -> int:
    assert qubit_to_int(qubit) == expected_result