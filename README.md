# nanograd

[![Language](https://img.shields.io/github/languages/top/zackxzhang/nanograd)](https://github.com/zackxzhang/nanograd)
[![Python](https://img.shields.io/pypi/pyversions/nanograd)](https://www.python.org)
[![License](https://img.shields.io/github/license/zackxzhang/nanograd)](https://opensource.org/licenses/BSD-3-Clause)
[![Last Commit](https://img.shields.io/github/last-commit/zackxzhang/nanograd)](https://github.com/zackxzhang/nanograd)

**nanograd** is a conceptual implementation of automatic differentiation

- illustrate fundamental principles behind differentiable programming
    - **nanograd** resembles Torch (dynamic, global, object-oriented)
    - ... more than it resembles Jax (static, local, functional)
- trace gradients through the directed acyclic graph of tensors and operators
    - leaves: parameters (tensors with gradient) and variables (tensors without gradient)
    - branches: operators that compose parameters, variables and other operators


#### example
- linear regression [here](./ols.py)
```python
from nanograd import Variable, Parameter, Optimizer

...

# model
t = Variable (t, tag='t')
x = Variable (x, tag='x')
w = Parameter(w, tag='w')
y = x @ w

# optimizer
optim = Optimizer(alpha=0.1)

# training
for _ in range(50):
    loss = squared_error(y, t)
    trace(loss, zero)
    params: list[Parameter] = list()
    trace(loss, Back(params))
    grads = [param.grad for param in params]
    optim.step(params, grads)
```