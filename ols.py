import numpy as np                                                # type: ignore
from nanograd import (
    Variable, Parameter, squared_error,
    Optimizer, trace, zero, back,
)
from util import RandomSeed


# settings
K = 4    # number of features
N = 300  # number of examples
S = 50   # number of steps


# create dataset
with RandomSeed(3) as rng:
    x = np.hstack((np.ones((N, 1)), rng.normal(size=(N, K))))
    v = rng.normal(0.0, 1.0, size=(K+1, 1))
    w = rng.normal(0.0, 1.0, size=(K+1, 1))
    e = rng.normal(0.0, 0.2, size=(N, 1))
    t = x @ v + e


# define model
t = Variable (t, tag='t')
x = Variable (x, tag='x')
w = Parameter(w, tag='w')
y = x @ w


# make optimizer
optim = Optimizer(alpha=0.1)


# before training
print(f"truth:   {v.flatten()}")
print(f"random:  {w.val.flatten()}")


# training loop
for _ in range(S):
    loss = squared_error(y, t)
    print(f"loss={loss.val:.6f}")
    trace(loss, zero)
    params, grads = trace(loss, back)
    optim.step(params, grads)


# after training
print(f"truth:   {v.flatten()}")
print(f"learned: {w.val.flatten()}")
