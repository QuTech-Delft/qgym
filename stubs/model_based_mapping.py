import cmodule


class ModelBasedMapping(cmodule.CPPModelBasedMapping):
    """
    Python extension of C++ class for a mapping pass in the OpenQL compiler that uses a RL model to generate a mapping.

    See [here](http://www.swig.org/Doc4.0/Python.html#Python_directors) for realising this type of Cross language
    polymorphism.
    """

    def __init__(self):
        super().__init__(self)


