# OpenQL-Gym

**_Note: This library is still under development and might change severely._**

A Reinforcement Learning gym for training agents on problems that occur frequently in quantum compilation. This library
is focussed around the specific compiler known as [OpenQL](https://github.com/QuTech-Delft/OpenQL).

The gym supports two environments:
1. Initial mapping: the problem of mapping virtual to physical qubits
2. Scheduling: the problem of scheduling quantum gate operations such that hardware and commutation constraints are satisfied.

## Tutorial notebooks
We provide some tutorial notebooks to get to know Reinforcement Learning and this library. These notebooks are found in
the folder [notebooks](notebooks).

## Installation
Below, we describe several steps for installing

### Building and installing from source
To build a wheel from this source one can run the command below.
This will create a built wheel in a folder called `dist`.

_Make sure that `pip`, `setuptools`, `wheel` are up-to-date._
```commandline
python setup.py bdist_wheel
```

### Setting up the environment
Initially, make sure you have Python installed on your computer. The python version should be either 3.8 or 3.9, other
versions are currently not supported.

You can check your Python version with the command `python --version`(Windows)/`python3 --version`(Unix).

Subsequently, open a terminal inside the folder containing the notebooks and execute the following commands. This will
create a Python virtual environment and require the qgym package and its requirements in it.

Windows:
```commandline
python -m venv venv
.\venv\Script\activate
pip install --upgrade pip setuptools wheel
pip install .\dist\qgym-0.1.0a0-py3-none-any.whl[tutorial]
```

Unix:
```commandline
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install ./dist/qgym-0.1.0a0-py3-none-any.whl[tutorial]
```