"""The qgym package consist of gyms and tools used for reinforcement learning (RL)
environments in the Quantum domain. It's main purpose is to easily create RL
environments for the different passes of the OpenQL framework, by simply initializing an
environment class. This abstraction of the environment allows RL developers to develop
RL agents to improve the OpenQL framework, without requiring prior knowledge of OpenQL.


Example:
    We want to create an environment for the OpenQL pass of initial mapping for a system
    with a QPU topology of 3x3. Using the ``qgym`` package this becomes:

    .. code-block:: python

        from qgym.envs import InitialMapping
        env = InitialMapping(0.5, connection_grid_size=(3,3))

    We can then use the environment in the code block above to train a stable baseline
    RL agent using the following code:

    .. code-block:: python

        from stable_baselines3 import PPO
        model = PPO("MultiInputPolicy", env, verbose=1)
        model.learn(int(1e5))

"""
import typing

import qgym.envs as envs
import qgym.spaces as spaces
import qgym.templates as templates
import qgym.utils as utils

__version__ = "0.1.0a0"

# pylint: skip-file
#
# Everything below here is largely based on OpenQL (Apache 2.0 License, Copyright [2016]
# [Nader Khammassi & Imran Ashraf, QuTech, TU Delft]):
# https://github.com/QuTech-Delft/OpenQL/blob/develop/LICENSE
#
# For the original file see: https://github.com/QuTech-Delft/OpenQL/blob/develop/python/openql/__init__.py
#
# Changes were made by updating the typemap in the _fixup_swig_autodoc_type function.
# Next to this, the __all__ dict was made consistent with our work. Also the occurences
# of openql were renamed to qgym (also similar words)


# Before we can import the dynamic modules, we have to set the linker search
# path appropriately.
import os

ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
if ld_lib_path:
    ld_lib_path += ":"
os.environ["LD_LIBRARY_PATH"] = ld_lib_path + os.path.dirname(__file__)
del ld_lib_path, os


# Import the SWIG-generated module into ourselves.
from .qgym import *

# List of all the relevant SWIG-generated stuff, to avoid outputting docs for
# all the other garbage SWIG generates for internal use.
__all__ = ["map_program", "envs", "spaces", "templates", "utils"]


# Swig's autodoc thing is nice, because it saves typing out all the overload
# signatures of each function. But it doesn't play nicely with Sphinx out of the
# box: Sphinx expects multiple signature lines to have a \ at the end (for as
# far as it's supported at all, you need 3.x at least), and SWIG outputs
# C++-style types rather than even trying to convert the types to Python.
# Python's object model to the rescue: we can just monkey-patch the docstrings
# after the fact.
def _fixup_swig_autodoc_type(typ: str) -> str:
    typ = typ.split()[0].split("::")[-1]
    typ = {
        "string": "str",
        "size_t": "int",
        "double": "float",
        "mapss": "Dict[str, str]",
        "vectorp": "List[Pass]",
        "vectorui": "List[int]",
        "vectori": "List[int]",
        "vectord": "List[float]",
    }.get(typ, typ)
    return typ


def _fixup_swig_autodoc_signature(sig: str) -> str:
    try:

        # Parse the incoming SWIG autodoc signature.
        begin, rest = sig.split("(", maxsplit=1)
        args, return_type = rest.split(")", maxsplit=1)
        name = begin.strip()
        return_type = return_type.strip()
        if return_type:
            assert return_type.startswith("-> ")
            return_type = return_type[3:]
        spacing = " " * (len(begin) - len(name))

        # Fix argument type names and use Python syntax for signature.
        args_list = args.split(",")
        for i, arg in enumerate(args_list):
            if not arg:
                continue
            toks = arg.split()
            if toks[-1] == "self":
                args_list[i] = "self"
                continue
            arg_name = toks[-1].split("=")
            if len(arg_name) > 1:
                default_val = " = " + arg_name[1]
            else:
                default_val = ""
            args_list[i] = (
                arg_name[0] + ": " + _fixup_swig_autodoc_type(toks[0]) + default_val
            )

        args = ", ".join(args_list)

        # Fix return type name.
        if return_type:
            return_type = _fixup_swig_autodoc_type(return_type)
        else:
            return_type = "None"

        sig = spacing + name + "(" + args + ") -> " + return_type

    except (Exception, AssertionError):
        pass

    return sig


def _fixup_swig_autodoc(ob: typing.Any, keep_sig: bool, keep_docs: bool) -> None:
    try:
        lines: typing.List[str] = ob.__doc__.split("\n")
        new_lines = []
        state = 0
        for line in lines:
            if state == 0:
                if line.strip():
                    state = 1
                    if keep_sig:
                        new_lines.append(_fixup_swig_autodoc_signature(line))
            elif state == 1:
                if not line.strip():
                    state = 2
                elif keep_sig:
                    new_lines[-1] += " \\"
                    new_lines.append(_fixup_swig_autodoc_signature(line))
            elif keep_docs:
                new_lines.append(line)
        while new_lines and not new_lines[-1]:
            del new_lines[-1]
        ob.__doc__ = "\n".join(new_lines) + "\n"
    except Exception:
        pass


for ob in __all__:
    ob = globals()[ob]
    if type(ob) == type:  # type: ignore[comparison-overlap]
        for mem in dir(ob):
            if mem == "__init__":
                _fixup_swig_autodoc(getattr(ob, mem), True, False)
            elif not mem.startswith("_"):
                if isinstance(getattr(ob, mem), property):
                    _fixup_swig_autodoc(getattr(ob, mem), False, True)
                else:
                    _fixup_swig_autodoc(getattr(ob, mem), True, True)

    else:
        _fixup_swig_autodoc(ob, True, True)

del typing, _fixup_swig_autodoc_signature, _fixup_swig_autodoc, ob
