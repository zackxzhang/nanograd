import numpy as np                                                # type: ignore
from abc import ABC, abstractmethod
from functools import wraps


class VariableTagger:

    counter: int = 0

    @classmethod
    def tag(cls) -> str:
        cls.counter += 1
        return f'variable {cls.counter}'


class ParameterTagger:

    counter: int = 0

    @classmethod
    def tag(cls) -> str:
        cls.counter += 1
        return f'parameter {cls.counter}'


def verify_operands(operator):
    @wraps(operator)
    def wrapped_operator(self, operand):
        if isinstance(operand, Tensor):
            return operator(self, operand)
        elif isinstance(operand, (int, float, np.ndarray)):
            return operator(self, Variable(operand))
        else:
            raise ValueError(
                'a binary operator only accepts '
                'two tensors as operands.'
            )
    return wrapped_operator


class Tensor(ABC):

    grad: np.ndarray

    @property
    @abstractmethod
    def gradable(self) -> bool:
        pass

    @property
    @abstractmethod
    def val(self):
        pass

    def __len__(self) -> int:
        return len(self.val)

    @property
    def shape(self):
        return self.val.shape

    @verify_operands
    def __add__(self, other):
        return Plus(self, other)

    @verify_operands
    def __radd__(self, other):
        return Plus(other, self)

    @verify_operands
    def __sub__(self, other):
        return Minus(self, other)

    @verify_operands
    def __rsub__(self, other):
        return Minus(other, self)

    @verify_operands
    def __mul__(self, other):
        return Product(self, other)

    @verify_operands
    def __rmul__(self, other):
        return Product(other, self)

    @verify_operands
    def __matmul__(self, other):
        return MatMul(self, other)

    @verify_operands
    def __rmatmul__(self, other):
        return MatMul(other, self)

    @verify_operands
    def __truediv__(self, other):
        return Quotient(self, other)

    @verify_operands
    def __rtruediv__(self, other):
        return Quotient(other, self)

    def __pow__(self, exponent: int | float):
        if isinstance(exponent, (int, float)):
            return Power(self, exponent)
        else:
            raise ValueError(
                'exponentiation only accepts '
                'an integer or a float as exponent'
            )

    def __neg__(self):
        return self * (-1.)

    @property
    def T(self):
        return Transpose(self)


class Variable(Tensor):

    def __init__(
        self,
        val: int | float | np.ndarray,
        tag: str | None = None,
    ):
        self._val = val
        self._tag = tag if tag else VariableTagger.tag()

    def __repr__(self) -> str:
        return self._tag

    @property
    def gradable(self):
        return False

    @property
    def val(self):
        return self._val


class Parameter(Tensor):

    def __init__(
        self,
        val: np.ndarray,
        tag: str | None = None,
    ):
        self.grad = 0.
        self._val = val
        self._tag = tag if tag else ParameterTagger.tag()

    def __repr__(self) -> str:
        return self._tag

    @property
    def gradable(self):
        return True

    @property
    def val(self):
        return self._val


class Operator(Tensor):

    @abstractmethod
    def vjp(self, v: np.ndarray):
        pass


class UnaryOperator(Operator):

    def __init__(self, operand: Tensor):
        self._operand = operand

    @property
    def operand(self) -> Tensor:
        return self._operand

    @property
    def gradable(self) -> bool:
        if self.operand.gradable:
            return True
        else:
            return False


class BinaryOperator(Operator):

    def __init__(self, operand_1: Tensor, operand_2: Tensor):
        self._operand_1 = operand_1
        self._operand_2 = operand_2

    @property
    def operand_1(self) -> Tensor:
        return self._operand_1

    @property
    def operand_2(self) -> Tensor:
        return self._operand_2

    @property
    def gradable(self) -> bool:
        if self.operand_1.gradable or self.operand_2.gradable:
            return True
        else:
            return False


class Plus(BinaryOperator):

    def __init__(self, operand_1: Tensor, operand_2: Tensor):
        super().__init__(operand_1, operand_2)

    def __repr__(self) -> str:
        return f'({self.operand_1} + {self.operand_2})'

    @property
    def val(self):
        return self.operand_1.val + self.operand_2.val

    def vjp(self, v: np.ndarray):
        if self.operand_1.gradable:
            self.operand_1.grad += v
        if self.operand_2.gradable:
            self.operand_2.grad += v
        # todo: support broadcasting


class Minus(BinaryOperator):

    def __repr__(self) -> str:
        return f'({self.operand_1} - {self.operand_2})'

    @property
    def val(self):
        return self.operand_1.val - self.operand_2.val

    def vjp(self, v: np.ndarray):
        if self.operand_1.gradable:
            self.operand_1.grad += v
        if self.operand_2.gradable:
            self.operand_2.grad -= v
        # todo: support broadcasting


class Product(BinaryOperator):

    def __repr__(self) -> str:
        return f'({self.operand_1} * {self.operand_2})'

    @property
    def val(self):
        return self.operand_1.val * self.operand_2.val

    def vjp(self, v: np.ndarray):
        if self.operand_1.gradable:
            j = self.operand_2.val
            self.operand_1.grad += v * j
        if self.operand_2.gradable:
            j = self.operand_1.val
            self.operand_2.grad += v * j
        # todo: support broadcasting


class MatMul(BinaryOperator):

    def __repr__(self) -> str:
        return f'({self.operand_1} @ {self.operand_2})'

    @property
    def val(self):
        return self.operand_1.val @ self.operand_2.val

    def vjp(self, v: np.ndarray):
        if self.operand_1.gradable:
            j = self.operand_2.val
            self.operand_1.grad += v @ j.T
        if self.operand_2.gradable:
            j = self.operand_1.val
            self.operand_2.grad += j.T @ v


class Quotient(BinaryOperator):

    def __repr__(self) -> str:
        return f'({self.operand_1} / {self.operand_2})'

    @property
    def val(self):
        return self.operand_1.val / self.operand_2.val

    def vjp(self, v: np.ndarray):
        if self.operand_1.gradable:
            j = 1 / self.operand_2.val
            self.operand_1.grad += v * j
        if self.operand_2.gradable:
            j = - self.operand_1.val / self.operand_2.val ** 2
            self.operand_2.grad += v * j


class Power(UnaryOperator):

    def __init__(self, operand: Tensor, exponent: int | float):
        super().__init__(operand)
        self._exponent = exponent

    def __repr__(self) -> str:
        return f'({self.operand}^{self.exponent})'

    @property
    def exponent(self):
        return self._exponent

    @property
    def val(self):
        return self.operand.val ** self.exponent

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            j = self.exponent * self.operand.val ** (self.exponent - 1)
            self.operand.grad += v * j


class Transpose(UnaryOperator):

    def __repr__(self) -> str:
        return f'{self.operand}.T'

    @property
    def val(self):
        return self.operand.val.T

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            self.operand.grad += v.T


class Absolute(UnaryOperator):

    def __repr__(self) -> str:
        return f'|{self.operand}|'

    @property
    def val(self):
        return np.abs(self.operand.val)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            self.operand.grad += v * np.sign(self.operand.val)


def _dims(ndim: int, axis: int, n: int):
    out = [1] * ndim
    out[axis] = n
    return out


class Summation(UnaryOperator):

    def __init__(self, operand: Tensor, axis: int):
        super().__init__(operand)
        self.axis = axis

    def __repr__(self) -> str:
        return f'Σ{self.operand}'

    @property
    def val(self):
        return np.sum(self.operand.val, axis=self.axis, keepdims=False)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            d = self.operand.val.ndim
            n = self.operand.val.shape[self.axis]
            self.operand.grad += np.tile(v, _dims(d, self.axis, n))


class Mean(UnaryOperator):

    def __init__(self, operand: Tensor, axis: int):
        super().__init__(operand)
        self.axis = axis

    def __repr__(self) -> str:
        return f'mean({self.operand})'

    @property
    def val(self):
        return np.mean(self.operand.val, axis=self.axis, keepdims=False)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            d = self.operand.val.ndim
            n = self.operand.val.shape[self.axis]
            self.operand.grad += np.tile(v / n, _dims(d, self.axis, n))


class Item(UnaryOperator):

    def __repr__(self) -> str:
        return f'({self.operand})'

    @property
    def val(self):
        return self.operand.val.item()

    def vjp(self, v: float):
        if self.operand.gradable:
            self.operand.grad += v * np.ones_like(self.operand.val, dtype=float)


class Logarithm(UnaryOperator):

    def __repr__(self) -> str:
        return f'log({self.operand})'

    @property
    def val(self):
        return np.log(self.operand.val)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            j = 1 / self.operand.val
            self.operand.grad += v * j


class Exponential(UnaryOperator):

    def __repr__(self) -> str:
        return f'exp({self.operand})'

    @property
    def val(self):
        return np.exp(self.operand.val)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            j = self.val
            self.operand.grad += v * j


class ReLU(UnaryOperator):

    def __repr__(self) -> str:
        return f'relu({self.operand})'

    @property
    def val(self):
        return np.maximum(self.operand.val, 0.)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            j = self.operand.val > 0.
            self.operand.grad += v * j


class TanH(UnaryOperator):

    def __repr__(self) -> str:
        return f'tanh({self.operand})'

    @property
    def val(self):
        return np.tanh(self.operand.val)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            j = 1 - self.val ** 2
            self.operand.grad += v * j


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Sigmoid(UnaryOperator):

    def __repr__(self) -> str:
        return f'sigmoid({self.operand})'

    @property
    def val(self):
        return _sigmoid(self.operand.val)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            s = self.val
            j = s * (1 - s)
            self.operand.grad += v * j


def _softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / np.sum(z, axis=-1, keepdims=True)


class Softmax(UnaryOperator):

    def __repr__(self) -> str:
        return f'softmax({self.operand})'

    @property
    def val(self):
        return _softmax(self.operand.val)

    def vjp(self, v: np.ndarray):
        if self.operand.gradable:
            n = len(self.operand)
            s = self.val
            diag_s = np.einsum('bi,ij->bij', s, np.eye(n))
            outer_s = np.einsum('bi,bj->bij', s, s)
            j = diag_s - outer_s
            self.operand.grad += v * j


def absolute(tensor: Tensor) -> Operator:
    return Absolute(tensor)


def summation(tensor: Tensor, axis: int = 0) -> Operator:
    return Summation(tensor, axis=axis)


def mean(tensor: Tensor, axis: int = 0) -> Operator:
    return Mean(tensor, axis=axis)


def item(tensor: Tensor) -> Operator:
    return Item(tensor)


def exp(tensor: Tensor) -> Operator:
    return Exponential(tensor)


def log(tensor: Tensor) -> Operator:
    return Logarithm(tensor)


def relu(tensor: Tensor) -> Operator:
    return ReLU(tensor)


def tanh(tensor: Tensor) -> Operator:
    return TanH(tensor)


def sigmoid(tensor: Tensor) -> Operator:
    return Sigmoid(tensor)


def softmax(tensor: Tensor) -> Operator:
    return Softmax(tensor)
