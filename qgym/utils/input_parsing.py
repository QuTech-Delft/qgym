"""This module contains function which parse user input.

With parsing we mean that the user input is validated and transformed to a predictable
format. In this way, user can give different input formats, but internally we are 
assured that the data has the same format."""
from copy import deepcopy
from typing import Type, Union

from qgym.templates import Rewarder
from qgym.utils.input_validation import check_instance


def parse_rewarder(
    rewarder: Union[Rewarder, None], default: Type[Rewarder]
) -> Rewarder:
    """Parse a `rewarder` given by the user.

    :param rewarder: ``Rewarder`` to use for the environment. If ``None``, then a new
        instance of the `default` rewarder will be returned.
    :param default: Type of the desired default rewarder to used when no rewarder is
        given.
    :return: A deepcopy of the given `rewarder` or a new instance of type `default` if
        `rewarder` is ``None``.
    """
    if rewarder is None:
        return default()
    check_instance(rewarder, "rewarder", Rewarder)
    return deepcopy(rewarder)
