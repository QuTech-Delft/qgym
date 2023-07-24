# Contributing
Currently, we are not looking for any new contributors. We are happy to receive any issues one encounters.

# Developer notes
Below is a list of useful information for developers.

### Installing
Install `qgym` in editable mode (preferably virtual) environment as follows:

```commandline
pip install --upgrade pip
pip install -e .[dev]
```
This will install the `qgym` package together with all tools needed for developing.

### Building a wheel
To build a wheel from this source one can run the command below.
This will create a built wheel in a folder called `dist`. Make sure that `pip`, `setuptools`, `wheel` are up-to-date.
```commandline
python setup.py bdist_wheel
```

### Building documentation
To build the documentation one should run the command below.
```commandline
python docs_files/make_docs.py
```
The HTML documentation will be placed in a folder called `docs`. This folder includes a
file `index.html`, open this in a browser to view the documentation.

To successfully build the documentation, one has to have the `qgym` packages installed.

### Developing jupyter notebooks
To launch a jupyter notebook environment ensure that the latest version of this library is installed.
If you are sure everything is up to date you can run:
```commandline
pip install -U .[dev]
jupyter notebook
```