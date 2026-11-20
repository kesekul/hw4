"""The module."""

from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
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


def _child_modules(value: object) -> list["Module"]:
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
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # weight Parameter
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )

        # bias Parameter
        self.bias = None
        if bias:
            bias_tensor = init.kaiming_uniform(
                out_features, 1, device=device, dtype=dtype
            )
            self.bias = Parameter(ops.reshape(bias_tensor, (out_features,)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)
        if self.bias is not None:
            output = ops.add(output, ops.broadcast_to(self.bias, output.shape))
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        total_dim = 1
        for dim in X.shape[1:]:
            total_dim = total_dim * dim
        return ops.reshape(X, (X.shape[0], total_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        labels_one_hot = init.one_hot(
            logits.shape[1], y, device=logits.device, dtype=logits.dtype
        )
        log_normalizer = ops.logsumexp(logits, axes=(1,))
        correct_scores = ops.summation(logits * labels_one_hot, axes=(1,))
        loss_vector = log_normalizer - correct_scores
        return ops.summation(loss_vector) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # broadcast w, b
        weight_broadcast = ops.broadcast_to(self.weight, x.shape)
        bias_broadcast = ops.broadcast_to(self.bias, x.shape)

        if self.training:
            # batch mean -> reshape -> broadcast
            mean_x = ops.divide_scalar(ops.summation(x, axes=(0,)), x.shape[0])
            mean_reshape = ops.reshape(mean_x, (1, mean_x.shape[0]))
            mean_broadcast = ops.broadcast_to(mean_reshape, x.shape)

            # var -> reshape -> broadcast
            var_inside = ops.power_scalar((x - mean_broadcast), 2)
            var_x = ops.divide_scalar(ops.summation(var_inside, axes=(0,)), x.shape[0])
            var_reshape = ops.reshape(var_x, (1, var_x.shape[0]))
            var_broadcast = ops.broadcast_to(var_reshape, x.shape)

            # batch norm
            top = x - mean_broadcast
            bottom = ops.power_scalar(ops.add_scalar(var_broadcast, self.eps), 0.5)
            total = ops.divide(top, bottom)

            # update running_mean and running_var
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean_x.detach()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var_x.detach()

            # multiply and add for trianing
            first_term = ops.multiply(weight_broadcast, total)
            return first_term + bias_broadcast

        # for evaluation
        else:
            # broadcast running_mean, running_var
            running_mean_broadcast = ops.broadcast_to(self.running_mean, x.shape)
            running_var_broadcast = ops.broadcast_to(self.running_var, x.shape)

            # normalize
            top_norm = x - running_mean_broadcast
            bottom_norm = ops.power_scalar(running_var_broadcast + self.eps, 0.5)
            final_norm = ops.divide(top_norm, bottom_norm)

            # apply weight and bias
            first_term = ops.multiply(weight_broadcast, final_norm)
            return first_term + bias_broadcast
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Any | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # mean -> reshape -> broadcast
        mean_x = ops.divide_scalar(ops.summation(x, axes=(1,)), x.shape[1])
        mean_reshape = ops.reshape(mean_x, (mean_x.shape[0], 1))
        mean_broadcast = ops.broadcast_to(mean_reshape, x.shape)

        # var -> reshape -> broadcast
        var_inside = ops.power_scalar((x - mean_broadcast), 2)
        var_sum = ops.summation(var_inside, axes=(1,))
        var_x = ops.divide_scalar(var_sum, x.shape[1])
        var_reshape = ops.reshape(var_x, (var_x.shape[0], 1))
        var_broadcast = ops.broadcast_to(var_reshape, x.shape)

        # parts of layer norm
        top = x - mean_broadcast
        bottom = ops.power_scalar(ops.add_scalar(var_broadcast, self.eps), 0.5)
        total = ops.divide(top, bottom)

        # broadcast w and b
        weight_broadcast = ops.broadcast_to(self.weight, x.shape)
        bias_broadcast = ops.broadcast_to(self.bias, x.shape)

        # multiply and add
        first_term = ops.multiply(weight_broadcast, total)
        return first_term + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            dropout_array = init.randb(
                *x.shape, p=1 - self.p, device=x.device, dtype=x.dtype
            )
            x_drop = ops.multiply(x, dropout_array)
            return x_drop / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
