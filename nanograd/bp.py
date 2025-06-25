import numpy as np                                                # type: ignore
from typing import Callable
from .op import (
    Tensor, Operator, UnaryOperator, BinaryOperator,
    Variable, Parameter, log, summation, mean, absolute, item,
)


class Optimizer:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def step(self, parameters: list[Parameter]):
        for param in parameters:
            param._val -= param.grad * self.alpha


def Back(parameters: list):
    def back(tensor: Tensor):
        if isinstance(op := tensor, Operator) and op.gradable:
            op.vjp(op.grad)
        if isinstance(pm := tensor, Parameter):
            parameters.append(pm)
    return back


def Zero(root: bool = True):
    def zero(tensor: Tensor):
        nonlocal root
        if root:
            tensor.grad = 1.
            root = False
        elif tensor.gradable:
            tensor.grad = 0.
    return zero


def Walk(nodes: list[str], edges: list[tuple[str, str]]):
    def walk(tensor: Tensor):
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
    return walk


def Punch(exits: dict, time: int = 0):
    def punch(tensor: Tensor):
        nonlocal time
        time += 1
        exits[tensor] = time
    return punch


def dfs(tensor: Tensor, visits: set, punch: Callable):
    if tensor in visits:
        return
    visits.add(tensor)
    if isinstance(tensor, (Parameter, Variable)):
        pass
    elif isinstance(tensor, UnaryOperator):
        dfs(tensor.operand, visits, punch)
    elif isinstance(tensor, BinaryOperator):
        dfs(tensor.operand_1, visits, punch)
        dfs(tensor.operand_2, visits, punch)
    else:
        raise TypeError(f"unexpected type {type(tensor)} value {tensor}")
    punch(tensor)


def toposort(tensor: Tensor):
    visits: set[Tensor] = set()
    exits: dict[Tensor, int] = dict()
    punch = Punch(exits)
    dfs(tensor, visits, punch)
    topo = sorted(visits, key=exits.__getitem__, reverse=True)
    return topo


def bfs(tensor: Tensor):
    visits: set[Tensor] = set()
    q = [tensor]
    while q:
        t = q.pop(0)
        if t in visits:
            continue
        if isinstance(t, (Variable, Parameter)):
            pass
        if isinstance(t, UnaryOperator):
            q.append(t.operand)
        elif isinstance(t, BinaryOperator):
            q.append(t.operand_1)
            q.append(t.operand_2)
        else:
            raise TypeError(f"unexpected type {type(t)} value {t}")


def trace(tensor: Tensor, hook: Callable):
    tensors = toposort(tensor)
    for t in tensors:
        hook(t)


class Loss(UnaryOperator):

    @property
    def val(self):
        return self.operand.val

    def vjp(self, v: float):
        if self.operand.gradable:
            self.operand.grad += v


class SquaredError(Loss):

    def __repr__(self) -> str:
        return f"({self.y} - {self.t})^2"

    def __init__(self, prediction: Operator, target: Variable):
        self.y = prediction
        self.t = target if isinstance(target, Variable) else Variable(target)
        super().__init__(item(mean((self.t - self.y) ** 2, axis=0)))


class CrossEntropy(Loss):

    def __repr__(self) -> str:
        return f"CrossEntropy({self.y}, {self.t})"

    def __init__(
        self,
        prediction: Operator,
        target: Variable,
        multiclass: bool = False,
    ):
        self.y = prediction
        self.t = target if isinstance(target, Variable) else Variable(target)
        if multiclass:  # one-hot vector encoding
            op = - item( mean(
                summation(self.t * log(self.y), axis=-1),
                axis=0,
            ) )
        else:  # 0-1 scalar encoding
            op = - item( mean(
                self.t * log(self.y) + (1. - self.t) * log(1. - self.y),
                axis=0,
            ) )
        super().__init__(op)


class Ridge(Loss):

    def __repr__(self) -> str:
        return f'({self.operand})'

    def __init__(self, parameter: Parameter):
        super().__init__(item(summation(parameter**2)))


class Lasso(Loss):

    def __repr__(self) -> str:
        return f'({self.operand})'

    def __init__(self, parameter: Parameter):
        super().__init__(item(summation(absolute(parameter))))


def squared_error(prediction: Operator, target: Variable):
    return SquaredError(prediction, target)


def cross_entropy(
    prediction: Operator,
    target: Variable,
    multiclass: bool = False,
):
    return CrossEntropy(prediction, target, multiclass=multiclass)


def ridge(parameter: Parameter):
    return Ridge(parameter)


def lasso(parameter: Parameter):
    return Lasso(parameter)
