from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axis = Z.ndim - 1
        maximum = Z.max(axis=axis, keepdims=True)
        shifted = Z - array_api.broadcast_to(maximum, Z.shape)
        log_total = array_api.log(array_api.exp(shifted).sum(axis=axis, keepdims=True))
        return shifted - array_api.broadcast_to(log_total, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        axis = len(node.shape) - 1
        keep_shape = node.shape[:-1] + (1,)
        total_grad = summation(out_grad, axes=(axis,)).reshape(keep_shape)
        return out_grad - exp(node) * total_grad.broadcast_to(node.shape)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[Union[int, tuple[int, ...]]] = None) -> None:
        ### BEGIN YOUR SOLUTION
        self.axes = (axes,) if isinstance(axes, int) else axes
        ### END YOUR SOLUTION

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        axes = tuple(range(Z.ndim)) if self.axes is None else self.axes
        axes = tuple(range(Z.ndim)[axis] for axis in axes)
        if not axes:
            return Z
        if len(set(axes)) != len(axes):
            raise ValueError("LogSumExp axes must be unique")

        maximum = Z
        for axis in axes:
            maximum = maximum.max(axis=axis, keepdims=True)

        total = array_api.exp(Z - array_api.broadcast_to(maximum, Z.shape))
        for axis in axes:
            total = total.sum(axis=axis, keepdims=True)

        out_shape = tuple(size for axis, size in enumerate(Z.shape) if axis not in axes)
        return (array_api.log(total) + maximum).reshape(out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        (Z,) = node.inputs
        ndim = len(Z.shape)
        axes = tuple(range(ndim)) if self.axes is None else self.axes
        axes = tuple(range(ndim)[axis] for axis in axes)
        if not axes:
            return out_grad
        keep_shape = tuple(
            1 if axis in axes else size for axis, size in enumerate(Z.shape)
        )

        # Normalize before restoring the maximum so its magnitude cannot erase the log-sum.
        maximum = Z.realize_cached_data()
        for axis in axes:
            maximum = maximum.max(axis=axis, keepdims=True)
        shift = Tensor.make_const(maximum).broadcast_to(Z.shape)
        weights = exp(Z - shift)
        total = summation(weights, axes=axes).reshape(keep_shape).broadcast_to(Z.shape)
        upstream = out_grad.reshape(keep_shape).broadcast_to(Z.shape)
        return upstream * (weights / total)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[Union[int, tuple[int, ...]]] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
