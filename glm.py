import numpy as np                                                # type: ignore
from nanograd import (
    Variable, Parameter, sigmoid, cross_entropy,
    Optimizer, trace, Zero, Back, ConstantSchedule,
)
from util import RandomSeed


# settings
K = 4     # number of features
N = 2000  # number of examples
S = 800   # number of steps


# create dataset
with RandomSeed(1) as rng:
    x = np.hstack((np.ones((N, 1)), rng.normal(size=(N, K))))
    v = rng.normal(0.0, 1.0, size=(K+1, 1))
    w = rng.normal(0.0, 1.0, size=(K+1, 1))
    e = rng.logistic(0.0, 1.0, size=(N, 1))
    t = np.where(x @ v + e > 0, 1., 0.)


# define model
t = Variable (t, tag='t')
x = Variable (x, tag='x')
w = Parameter(w, tag='w')
y = sigmoid(x @ w)


# make optimizer
optim = Optimizer(alpha=ConstantSchedule(0.2))


# before training
print(f"truth:   {v.flatten()}")
print(f"random:  {w.val.flatten()}")


# training loop
for _ in range(S):
    loss = cross_entropy(y, t)
    print(f"loss={loss.val:.6f}")
    trace(loss, Zero())
    params: list[Parameter] = list()
    trace(loss, Back(params))
    optim.step(params)


# after training
print(f"truth:   {v.flatten()}")
print(f"learned: {w.val.flatten()}")
