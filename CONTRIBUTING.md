# Contributing
Currently, we are not looking for any new contributors. We are happy to receive any issues one encounters.

# Developer notes
Below is a list of useful information for developers.


## Install in developer mode
1. Clone the project and move to the folder.
    ```commandline
    git clone --recurse-submodules https://github.com/qutech-sd/OpenQL-Gym.git
    cd OpenQl-Gym
    ```
1. Make sure `wheel` and `plumbum` are up to date.
    ```commandline
    pip install --upgrade wheel plumbum
    ```
1. Instal in editable mode with developer options (takes $\pm$ 15 minutes).
    ```commandline
    pip install -v -e .[dev]

You might also require certain Linux packages when building, at least `gcc`, `flex >2.6`, `bison > 3.0`, `cmake >= 3.0`,
and `swig >=3.0.12` can be required.
See also [the OpenQL dev docs](https://openql.readthedocs.io/en/latest/developer/build.html) for more info.


## Building Python wheels
The following steps will create a python wheel in the folder `pybuild/dist`.

1. Clone the project and move to the folder.
    ```commandline
    git clone --recurse-submodules https://github.com/qutech-sd/OpenQL-Gym.git
    cd OpenQl-Gym
    ```
1. Make sure `wheel` and `plumbum` are up to date.
    ```commandline
    pip install --upgrade wheel plumbum
    ```
1. Instal in editable mode with developer options (takes $\pm$ 15 minutes).
    ```commandline
    python setup.py bdist_wheel
    ```


## Building documentation
To build the documentation one should run the two commands below.
The HTML documentation will be placed in a folder called `docs_html`. This folder includes a file `index.html`, open this
in a browser to view the documentation.

To successfully build the documentation it is required to have the `qgym` package compiled and installed in the environment.
```commandline
sphinx-apidoc -o docs_build -f -F -M -e -t docs --implicit-namespaces pybuild/qgym
sphinx-build docs_build docs_html
```

## Developing jupyter notebooks
To launch a jupyter notebook environment ensure that the latest version of this library is installed,
including the developer dependencies.

_Make sure that `pip`, `setuptools`, `wheel` are up-to-date (see [above](#installing-requirements))_
```commandline
pip install -U .[dev]
jupyter notebook
```