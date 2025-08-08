import numpy as np                                                # type: ignore
from util import RandomSeed
from nanograd import (
    Tensor, Variable, Parameter,
    sigmoid, tanh, softmax, cross_entropy,
    Optimizer, trace, Zero, Back, CosineSchedule,
)
from nanograd.op import _sigmoid, _softmax


# recurrent network settings
T = 7     # time steps
I = 112   # input pixels
O = 10    # output classes
H1 = 256  # hidden dimension (layer 1)
H2 = 256  # hidden dimension (layer 2)


# recurrent network
with RandomSeed(2) as rng:
    weights = {
        'state': {
            'h1': Parameter(rng.normal(size=(1, H1))),
            'h2': Parameter(rng.normal(size=(1, H2))),
        },
        'layer1': {
            'Wz': Parameter(rng.normal(size=(I, H1))),
            'Wr': Parameter(rng.normal(size=(I, H1))),
            'Wh': Parameter(rng.normal(size=(I, H1))),
            'Uz': Parameter(rng.normal(size=(H1, H1))),
            'Ur': Parameter(rng.normal(size=(H1, H1))),
            'Uh': Parameter(rng.normal(size=(H1, H1))),
            'bz': Parameter(np.zeros((1, H1))),
            'br': Parameter(np.zeros((1, H1))),
            'bh': Parameter(np.zeros((1, H1))),
        },
        'layer2': {
            'Wz': Parameter(rng.normal(size=(H1, H2))),
            'Wr': Parameter(rng.normal(size=(H1, H2))),
            'Wh': Parameter(rng.normal(size=(H1, H2))),
            'Uz': Parameter(rng.normal(size=(H2, H2))),
            'Ur': Parameter(rng.normal(size=(H2, H2))),
            'Uh': Parameter(rng.normal(size=(H2, H2))),
            'bz': Parameter(np.zeros((1, H2))),
            'br': Parameter(np.zeros((1, H2))),
            'bh': Parameter(np.zeros((1, H2))),
        },
        'head': {
            'W': Parameter(rng.normal(size=(H2, O)) * 0.1),
            'b': Parameter(np.zeros((1, O))),
        },
    }


def broadcast(x: Tensor | np.ndarray):
    if isinstance(x, Tensor):
        n = x.val.shape[0]
    else:
        n = x.shape[0]
    return Variable(np.ones((n, 1)))


def gru(x, e, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh):
    z = sigmoid(x @ Wz + h @ Uz + e @ bz)
    r = sigmoid(x @ Wr + h @ Ur + e @ br)
    c = tanh(x @ Wh + (r * h) @ Uh + e @ bh)
    return (1. - z) * h + z * c


def clf(x, e, W, b):
    logits = x @ W + e @ b
    probs = softmax(logits)
    return probs


def rnn(X, weights):
    assert isinstance(X, Variable)
    h1, h2 = weights['state']['h1'], weights['state']['h2'],
    e = broadcast(X)
    h1 = e @ h1
    h2 = e @ h2
    for t in range(T):
        x = X[:, t]
        p1 = weights['layer1']
        h1 = gru(
            x, e, h1,
            p1['Wz'], p1['Uz'], p1['bz'],
            p1['Wr'], p1['Ur'], p1['br'],
            p1['Wh'], p1['Uh'], p1['bh']
        )
        p2 = weights['layer2']
        h2 = gru(
            h1, e, h2,
            p2['Wz'], p2['Uz'], p2['bz'],
            p2['Wr'], p2['Ur'], p2['br'],
            p2['Wh'], p2['Uh'], p2['bh']
        )
    ph = weights['head']
    y = clf(h2, e, ph['W'], ph['b'])
    return y


# data & training settings
mnist = np.load('data/mnist.npz')
X = mnist['x_train']  # (batch, time, feature)
Y = mnist['y_train']  # (batch, class)
N = len(X)  # data size
n = 64      # batch size
S = 50_000  # training steps
optim = Optimizer(alpha=CosineSchedule(S, 1e-2, 1e-4))


# training
with RandomSeed(4) as rng:
    for s in range(S):
        idx = rng.choice(N, n, replace=False)
        x, t = Variable(X[idx]), Variable(Y[idx])
        y = rnn(x, weights)
        loss = cross_entropy(y, t, multiclass=True)
        print(f"loss={loss.val:.6f}")
        trace(loss, Zero())
        params: list[Parameter] = list()
        trace(loss, Back(params))
        optim.step(params)


# testing settings
X = mnist['x_test']
Y = mnist['y_test']
N = len(X)  # data size
n = 200     # batch size


# testing
c = 0  # number of correct predictions
i = 0  # pointer to the test examples
while i < N:
    x = Variable(X[i:i+n])
    t = Y[i:i+n]
    y = rnn(x, weights).val
    c += np.sum(np.argmax(t, axis=1) == np.argmax(y, axis=1))
    i += n
print(f"accuracy={c / N:.2%}")
