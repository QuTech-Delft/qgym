from typing import Callable

import cmodule  # this should be the OpenQL lib


class ModelBasedMapping(cmodule.CPPModelBasedMapping):
    """
    Python extension of C++ class for a mapping pass in the OpenQL compiler that uses a RL model to generate a mapping.

    See [here](http://www.swig.org/Doc4.0/Python.html#Python_directors) for realising this type of Cross language
    polymorphism.
    """

    def __init__(self) -> None:
        super().__init__(self)

    def set_model(self, model: Callable[]) -> None:
        """
        Set RL-based model that on input of a state returns an action.
        :return:
        """
        done = False
        while not done:
            action = model(state)
            done = state.do_action(action)


