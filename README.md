[![CI](https://github.com/davidfitzek/qflow/actions/workflows/python-package.yml/badge.svg)](https://github.com/davidfitzek/qflow/actions/workflows/python-package.yml)


# A library for variational quantum algorithms

## Installation and use
The package `qflow` is prepared such that it can be installed in development mode locally (such that any change in the code is instantly reflected). To install `qflow` locally, clone/download the repository and run the following command from the qflow folder:

```
pip install -e .
```

Then, run any of the notebooks in the example folder to play around with variational quantum circuits. 

Feel free to reach out to me at dpfitzek@gmail.com if you face any issues.

For testing and verification of the software run:

```
python -m pytest src/qflow/tests
```

If you want to deinstall the library use:

```
pip uninstall qflow
```

To run the quantum chemistry examples you need to install pyscf. Before you can install this package make sure that cmake is installed on your device. If you are using Mac run 

```
brew install cmake
brew install llvm
```


## Background



## References
