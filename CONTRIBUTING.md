# Contributing
Currently, we are not looking for any new contributors. We are happy to receive any issues one encounters.

# Developer notes
Below is a list of useful information for developers.

### Installing requirements
Make sure to install all requirements in your (preferably virtual) environment as follows.
The first command installs the latest versions of `pip`, `setuptools`, and `wheel`.
```commandline
pip install -U pip setuptools wheel
pip install -U -r requirements.txt
```

### Building and installing from source
To build a wheel from this source one can run the command below.
This will create a built wheel in a folder called `dist`.

_Make sure that `pip`, `setuptools`, `wheel` are up-to-date (see [above](#installing-requirements))_
```commandline
python setup.py bdist_wheel
```

### Building documentation
To build the documentation one should run the two commands below.
The HTML documentation will be placed in a folder called `docs`. This folder includes a
file `index.html`, open this in a browser to view the documentation.

To successfully build the documentation, one has to have the `qgym` packages installed
in the environment.
```commandline
python docs_files/make_docs.py
```

### Developing jupyter notebooks
To launch a jupyter notebook environment ensure that the latest version of this library is installed,
including the developer dependencies.

_Make sure that `pip`, `setuptools`, `wheel` are up-to-date (see [above](#installing-requirements))_
```commandline
pip install -U .[dev]
jupyter notebook
```