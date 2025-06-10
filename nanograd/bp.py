import numpy as np                                                # type: ignore
from typing import Callable
from .op import (
    Tensor, Operator, UnaryOperator, BinaryOperator,
    Variable, Parameter, log
)


class Optimizer:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def step(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param._val -= grad * self.alpha


def back(tensor: Tensor):
    if isinstance(op := tensor, Operator) and op.gradable:
        op.vjp(op.grad)


def zero(tensor: Tensor):
    if tensor.gradable:
        tensor.grad = 0.
    if isinstance(loss := tensor, Loss):
        loss.grad = 1.


def leaf(tensor: Tensor):
    if isinstance(tensor, Parameter):
        print(f"parameter: {tensor}")
    elif isinstance(tensor, Variable):
        print(f"variable: {tensor}")


def walk(nodes: list[str], edges: list[tuple[str, str]]):
    def tape(tensor: Tensor):
        if isinstance(tensor, (Parameter, Variable)):
            nodes.append(str(tensor))
        elif isinstance(tensor, UnaryOperator):
            nodes.append(str(tensor))
            edges.append((str(tensor), str(tensor.operand)))
        elif isinstance(tensor, BinaryOperator):
            nodes.append(str(tensor))
            edges.append((str(tensor), str(tensor.operand_1)))
            edges.append((str(tensor), str(tensor.operand_2)))
        else:
            raise TypeError(f"unexpected type {type(tensor)} value {tensor}")
        return nodes, edges
    return tape


def trace(tensor: Tensor, hook: Callable):
    parameters = list()
    gradients  = list()
    q = [tensor]
    while q:
        t = q.pop(0)
        if isinstance(t, UnaryOperator):
            hook(t)
            q.append(t.operand)
        elif isinstance(t, BinaryOperator):
            hook(t)
            q.append(t.operand_1)
            q.append(t.operand_2)
        elif isinstance(t, Variable):
            hook(t)
        elif isinstance(t, Parameter):
            hook(t)
            parameters.append(t)
        else:
            raise TypeError(f"unexpected type {type(t)} value {t}")
    gradients = [param.grad for param in parameters]
    return parameters, gradients


class Loss(UnaryOperator):

    y: Operator
    t: Variable

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ +
            f"(prediction={self.y}, target={self.t})"
        )

    @property
    def val(self):
        return self.operand.val.item()

    def vjp(self, v: Tensor):
        if self.operand.gradable:
            self.operand.grad += np.array([[1.]])


class SquaredError(Loss):

    def __init__(self, prediction: Operator, target: Variable):
        self.y = prediction
        self.t = target
        n = self.t.val.shape[0]
        u = Variable(np.ones((1,n)))
        op = u @ ((self.t - self.y) ** 2) * (1/n)
        super().__init__(op)


class CrossEntropy(Loss):

    def __init__(self, prediction: Operator, target: Variable):
        self.y = prediction
        self.t = target
        n = self.t.val.shape[0]
        u = Variable(np.ones((1,n)))
        op = u @ (self.t * log(self.y) + (1.-self.t) * log(1.-self.y)) * (-1./n)
        super().__init__(op)


def squared_error(prediction: Operator, target: Variable):
    return SquaredError(prediction, target)


def cross_entropy(prediction: Operator, target: Variable):
    return CrossEntropy(prediction, target)
