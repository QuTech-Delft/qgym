# QGYM – A Gym for Training and Benchmarking RL-Based Quantum Compilation
`qgym` is a software framework that provides environments for training and benchmarking RL-based quantum compilers.
It is built on top of OpenAI Gym and abstracts parts of the compilation process that are irrelevant to AI researchers.
`qgym` includes three environments: Initial Mapping, Qubit Routing, and Scheduling, each of which is customizable and extensible.

## Documentation
We have created an [extensive documentation](https://qutech-delft.github.io/qgym/index.html) with code snippets.
Please feel free to contact us via <s.feld@tudelft.nl> if you have any questions, or by creating a [GitHub issue](https://github.com/QuTech-Delft/qgym/issues/new).

## Getting Started
What follows are some simple steps to get you running.
You could also have a look at some [Jupyter Notebooks](https://github.com/QuTech-Delft/qgym/tree/master/notebooks) that we have created for a tutorial at the [IEEE International Conference on Quantum Computing and Engineering (QCE’22)](https://qce.quantum.ieee.org/2022/tutorials-program/).

### Installing with pip
To install the `qgym` use
```terminal
pip install qgym
```
If you would also like to use the notebooks, additional packages are required.
These can simply be installed by using
In this case, use
```terminal
pip install qgym[tutorial]
```

Currently `qgym` has support for Python 3.7, 3.8, 3.9, 3.10 and 3.11.


## Publication
The paper on `qgym` has been presented in the [1st International Workshop on Quantum Machine Learning: From Foundations to Applications (QML@QCE'23)](https://qml.lfdr.de/2023/).
You can find the preprint of the paper on [arxiv](https://arxiv.org/pdf/2308.02536.pdf).

```terminal
@article{van2023qgym,
  title={qgym: A Gym for training and benchmarking RL-based quantum compilation},
  author={van der Linde, Stan and de Kok, Willem and Bontekoe, Tariq and Feld, Sebastian},
  journal={arXiv preprint arXiv:2308.02536},
  year={2023}
}
```
## Team
Building qgym is a joint effort.

### Core developers
- Stan van der Linde
- Will de Kok
- Tariq Bontekoe
- Sebastian Feld

### Power users
- Joris Henstra
- Rares Oancea 
