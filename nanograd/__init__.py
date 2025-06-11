from .op import Variable, Parameter
from .bp import Optimizer, trace, zero, Back, leaf, Walk
from .fn import (
    exp, log, relu, tanh, sigmoid, _sigmoid, softmax, _softmax,
    squared_error, cross_entropy,
)


__version__ = '0.1.2'
