# nanograd

[![Language](https://img.shields.io/github/languages/top/zackxzhang/nanograd)](https://github.com/zackxzhang/nanograd)
[![Python](https://img.shields.io/pypi/pyversions/nanograd)](https://www.python.org)
[![License](https://img.shields.io/github/license/zackxzhang/nanograd)](https://opensource.org/licenses/BSD-3-Clause)
[![Last Commit](https://img.shields.io/github/last-commit/zackxzhang/nanograd)](https://github.com/zackxzhang/nanograd)

A conceptual implementation of autograd

- illustrate the fundamental principle behind automatic differentiation
- trace gradients through the directed acyclic graph of tensors and operators
    - leaves: parameters (tensors with gradient) and variables (tensors without gradient)
    - branches: operators that compose parameters, variables and other operators
