# Contributing
Currently, we are not looking for any new contributors. We are happy to receive any issues one encounters.

# Developer notes
Below is a list of useful information for developers.

### TODO: IMPROVE THIS FOR BUILD WITH CPP CODE, BUT THIS WORKS FOR LINUX
But basically you need `wheel` and `plumbum` installed and can then run either of the following:
```commandline
pip install -v .
pip install -v -e .
```
We advise the editable install (second line) when developing.

You might also require certain Linux packages when building, at least `gcc`, `flex >2.6`, `bison > 3.0`, `cmake >= 3.0`,
and `swig >=3.0.12` can be required. See also [the OpenQL dev docs](https://openql.readthedocs.io/en/latest/developer/build.html)
for more info.

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
The HTML documentation will be placed in a folder called `docs_html`. This folder includes a file `index.html`, open this
in a browser to view the documentation.

To successfully build the documentation, one has to have the `qgym` package compiled and installed in the environment.
```commandline
sphinx-apidoc -o docs_build -f -F -M -e -t docs --implicit-namespaces pybuild/qgym
sphinx-build docs_build docs_html
```

### Developing jupyter notebooks
To launch a jupyter notebook environment ensure that the latest version of this library is installed,
including the developer dependencies.

_Make sure that `pip`, `setuptools`, `wheel` are up-to-date (see [above](#installing-requirements))_
```commandline
pip install -U .[dev]
jupyter notebook
```