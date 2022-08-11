# OpenQL-Gym - notebook tutorial
Below we explain how to prepare your environment for using the notebooks.

## Setting up the environment
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
