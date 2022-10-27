# OpenQL-Gym - notebook tutorial

## Easy mode (binder)
Follow the links as stated in the table to get a running notebook instance.

| Name | Binder notebook                                                                                        | Binder solution notebook                                                                                            |
|---|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| 1. Maze runner | [link](https://mybinder.org/v2/gh/qutech-sd/OpenQL-Gym/notebooks/main?labpath=EMPTY_maze_runner.ipynb)           | [link](https://mybinder.org/v2/gh/qutech-sd/OpenQL-Gym/notebooks?labpath=COMPLETED_maze_runner.ipynb)               |
| 2. Mapping qubits | [link](https://mybinder.org/v2/gh/qutech-sd/OpenQL-Gym/notebooks/main?labpath=EMPTY_mapping_qubits.ipynb)        | [link](https://mybinder.org/v2/gh/qutech-sd/OpenQL-Gym/notebooks?labpath=COMPLETED_mapping_qubits.ipynb)        |
| 3. Scheduling Operations | [link](https://mybinder.org/v2/gh/qutech-sd/OpenQL-Gym/notebooks/main?labpath=EMPTY_scheduling_operations.ipynb) | [link](https://mybinder.org/v2/gh/qutech-sd/OpenQL-Gym/notebooks?labpath=COMPLETED_scheduling_operations.ipynb) |

## Manual deployment
Below we explain how to prepare an environment locally on your device for using the notebooks. One should download all the files in this repository on order to run this locally.

### Setting up the environment
Initially, make sure you have [Python](https://www.python.org/downloads/) installed on your computer and that `python` 
is on your path (can be verified by running `python` in a command window).

If this is not the case for an existing installation one can add `<path to python folder>` and
`<path to python folder>\Scripts` to the `PATH` environment variable. For a new installation remember to check 
"Add Python to environment variables" (should be checked by default).

The Python version should be either [3.8](https://www.python.org/downloads/release/python-3810/) or
[3.9](https://www.python.org/downloads/release/python-3913/), other versions are currently not supported.

You can check your Python version with the command `python --version`(Windows)/`python3 --version`(Unix).

Subsequently, open a terminal inside the folder containing the notebooks and this README and execute the following
commands. This will create a Python virtual environment and require the qgym package and its requirements in it.

Windows:
```commandline
python -m venv venv
.\venv\Script\activate
pip install --upgrade pip setuptools wheel
pip install .\qgym-0.1.0a0-py3-none-any.whl[tutorial]
```

Unix:
```commandline
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install ./qgym-0.1.0a0-py3-none-any.whl[tutorial]
```

Finally, one starts a Jupyter notebook server by running the following command:

```commandline
jupyter notebook
```

As a result, either your browser will open a new webpage automatically, or you should copy the provided link into a
browser.

### Documentation
If you're interested in more details on the code, one can find a (hopefully) up-to-date version of the documentation can
by opening `docs/index.html` in your browser.
