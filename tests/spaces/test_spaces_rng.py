from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from qgym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gymnasium import Space


@pytest.mark.parametrize(
    ("space", "args"),
    [
        (Box, (0, 10, (10, 10))),
        (Discrete, (10,)),
        (MultiDiscrete, ([10] * 10,)),
        (MultiBinary, (100,)),
    ],
)
def test_init_rng(space: type[Space[Any]], args: Iterable[Any]) -> None:
    space1 = space(*args, rng=np.random.default_rng(0))  # type: ignore[call-arg]
    space2 = space(*args, rng=np.random.default_rng(0))  # type: ignore[call-arg]
    space3 = space(*args, rng=np.random.default_rng(1))  # type: ignore[call-arg]

    np.testing.assert_array_equal(space1.sample(), space2.sample())
    assert not np.array_equal(space1.sample(), space3.sample())


def test_init_rng_dict() -> None:
    space1 = Dict({"test": MultiBinary(100)}, rng=np.random.default_rng(0))
    space2 = Dict({"test": MultiBinary(100)}, rng=np.random.default_rng(0))
    space3 = Dict({"test": MultiBinary(100)}, rng=np.random.default_rng(1))

    np.testing.assert_array_equal(space1.sample()["test"], space2.sample()["test"])
    assert not np.array_equal(space1.sample()["test"], space3.sample()["test"])


@pytest.mark.parametrize(
    "space",
    [Box(0, 10, (10, 10)), Discrete(10), MultiDiscrete([10] * 10), MultiBinary(100)],
)
def test_seed(space: Space[Any]) -> None:
    space1 = deepcopy(space)
    space2 = deepcopy(space)
    space3 = deepcopy(space)

    space1.seed(0)
    space2.seed(0)
    space3.seed(1)

    np.testing.assert_array_equal(space1.sample(), space2.sample())
    assert not np.array_equal(space1.sample(), space3.sample())


def test_seed_dict() -> None:
    space1 = Dict({"test": MultiBinary(100)})
    space2 = Dict({"test": MultiBinary(100)})
    space3 = Dict({"test": MultiBinary(100)})

    space1.seed(0)
    space2.seed(0)
    space3.seed(1)

    np.testing.assert_array_equal(space1.sample()["test"], space2.sample()["test"])
    assert not np.array_equal(space1.sample()["test"], space3.sample()["test"])
