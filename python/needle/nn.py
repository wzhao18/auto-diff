"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features, fan_out=out_features, device=device, dtype=dtype, requires_grad=True
        ))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(
                fan_in=out_features, fan_out=1, device=device, dtype=dtype, requires_grad=True
            ).reshape((1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = X @ self.weight
        if self.bias:
            res = res + ops.broadcast_to(self.bias, res.shape)
        return res
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        num_features = math.prod(X.shape[1:])
        return ops.reshape(X, (batch_size, num_features))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for mod in self.modules:
            x = mod(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, num_features = logits.shape
        one_hot = init.one_hot(num_features, y)
        y_hat = ops.summation(logits * one_hot, (1,))
        loss = ops.summation(ops.logsumexp(logits, (1,)) - y_hat) / batch_size
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, dtype=dtype, device=device))
        self.running_mean = init.zeros(dim, dtype=dtype, device=device)
        self.running_var = init.ones(dim, dtype=dtype, device=device)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training:
            mean = x.sum((0,)) / batch_size
            mean_broad = mean.broadcast_to(x.shape)
            var = (((x - mean_broad) ** 2).sum((0,)) / batch_size)
            var_broad = var.broadcast_to(x.shape)
            normalized = (x - mean_broad) / ((var_broad + self.eps) ** 0.5)

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            run_mean_broad = self.running_mean.broadcast_to(x.shape)
            run_var_broad = ((self.running_var + self.eps) ** 0.5).broadcast_to(x.shape)
            normalized = (x - run_mean_broad) / run_var_broad

        output = self.weight.broadcast_to(x.shape) * normalized + (self.bias).broadcast_to(x.shape)
        return output
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, dtype=dtype, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_features = x.shape
        mean = (x.sum((1,)) / num_features).reshape((batch_size, 1)).broadcast_to(x.shape)
        var = (((x - mean) ** 2).sum((1,)) / num_features).reshape((batch_size, 1)).broadcast_to(x.shape)
        normalized = (x - mean) / ((var + self.eps) ** 0.5)
        output = self.weight.broadcast_to(x.shape) * normalized + self.bias.broadcast_to(x.shape)
        return output
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            keep_prob = 1 - self.p
            drop_mat = init.randb(*(x.shape), p=keep_prob)
            return (x * drop_mat) / keep_prob
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



