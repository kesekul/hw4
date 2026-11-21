"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        base, exponent = node.inputs
        partial_base = (out_grad * exponent * power(base, exponent - 1))
        partial_exp = (out_grad * power(base, exponent) * log(base))
        return partial_base, partial_exp
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.scalar == 0:
            return out_grad * 0
        a = node.inputs[0]
        a_partial = out_grad * self.scalar * (a ** (self.scalar - 1))
        return a_partial
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        divident, divisor = node.inputs
        divident_partial = out_grad * (power_scalar(divisor,-1))
        divisor_partial = out_grad * negate(divident) * power_scalar(divisor, -2)
        return (divident_partial, divisor_partial)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1/self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        first, second = (-1, -2) if self.axes is None else self.axes
        if BACKEND == "np":
            return array_api.swapaxes(a, first, second)
        order = list(range(a.ndim))
        order[first], order[second] = order[second], order[first]
        return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "nd":
            a = a.compact()
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "nd" and a.ndim < len(self.shape):
            leading = (1,) * (len(self.shape) - a.ndim)
            a = a.compact().reshape(leading + a.shape)
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape

        # Align the input shape with the right side of the output shape.
        extra_dims = len(output_shape) - len(input_shape)
        sum_axes = list(range(extra_dims))

        # An input dimension of size 1 was duplicated during broadcasting.
        for i, input_dim in enumerate(input_shape):
            output_axis = extra_dims + i
            if input_dim == 1 and output_shape[output_axis] != 1:
                sum_axes.append(output_axis)

        grad = out_grad

        if sum_axes:
            grad = summation(grad, axes=tuple(sum_axes))

        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return array_api.sum(a, axis=self.axes)
        if self.axes is None:
            return a.sum().reshape(())
        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        axes = tuple(range(a.ndim)[axis] for axis in axes)
        for axis in sorted(axes, reverse=True):
            a = a.sum(axis=axis)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        # Convert negative axes to their positive equivalents.
        axes = tuple(
            axis if axis >= 0 else axis + len(input_shape)
            for axis in axes
        )

        # Reinsert each dimension removed by the forward summation.
        reshape_shape = list(input_shape)

        for axis in axes:
            reshape_shape[axis] = 1

        grad = reshape(out_grad, tuple(reshape_shape))
        return broadcast_to(grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        grad_lhs = matmul(out_grad, transpose(rhs))
        grad_rhs = matmul(transpose(lhs), out_grad)

        def reduce_to_shape(grad, target_shape):
            grad_shape = grad.shape
            extra_dims = len(grad_shape) - len(target_shape)
            sum_axes = list(range(extra_dims))

            # Sum dimensions that were broadcast from size 1.
            for i, target_dim in enumerate(target_shape):
                grad_axis = extra_dims + i

                if target_dim == 1 and grad_shape[grad_axis] != 1:
                    sum_axes.append(grad_axis)

            if sum_axes:
                grad = summation(grad, axes=tuple(sum_axes))

            return reshape(grad, target_shape)

        grad_lhs = reduce_to_shape(grad_lhs, lhs.shape)
        grad_rhs = reduce_to_shape(grad_rhs, rhs.shape)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * -1
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * power_scalar(node.inputs[0],-1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = Tensor(
          node.realize_cached_data() > 0,
          device=out_grad.device,
          dtype=out_grad.dtype,
          requires_grad=False,
        )
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 + negate(node * node))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: tuple[NDArray, ...]) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return array_api.stack(args, axis=self.axis)
        if not args:
            raise ValueError("stack requires at least one tensor")
        first = args[0]
        axis = range(first.ndim + 1)[self.axis]
        shape = first.shape[:axis] + (len(args),) + first.shape[axis:]
        result = array_api.empty(shape, device=first.device)
        for index, item in enumerate(args):
            if item.shape != first.shape or item.device != first.device:
                raise ValueError("stack requires matching shapes and devices")
            selection = (slice(None),) * axis + (index,)
            selection += (slice(None),) * (first.ndim - axis)
            result[selection] = item
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis = range(A.ndim)[self.axis]
        if BACKEND == "np":
            return tuple(array_api.take(A, i, axis=axis) for i in range(A.shape[axis]))
        shape = A.shape[:axis] + A.shape[axis + 1:]
        pieces = []
        for index in range(A.shape[axis]):
            selection = (slice(None),) * axis + (index,)
            selection += (slice(None),) * (A.ndim - axis - 1)
            pieces.append(A[selection].compact().reshape(shape))
        return tuple(pieces)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int, output_shape=None):
        self.axes = axes
        self.dilation = dilation
        ### BEGIN YOUR SOLUTION
        # The adjoint of strided slicing must restore the original extent.
        self.output_shape = output_shape
        ### END YOUR SOLUTION

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # HW4 treats axes outside the input rank as no-ops.
        axes = {axis % a.ndim for axis in self.axes if -a.ndim <= axis < a.ndim}
        step = self.dilation + 1
        if step < 1:
            raise ValueError("dilation must be nonnegative")
        if step == 1 or not axes:
            return a
        shape = self.output_shape
        if shape is None:
            shape = tuple(size * step if axis in axes else size
                          for axis, size in enumerate(a.shape))
        selection = tuple(slice(None, None, step if axis in axes else 1)
                          for axis in range(a.ndim))
        if BACKEND == "np":
            result = array_api.zeros(shape, dtype=a.dtype)
        else:
            result = array_api.full(shape, 0, device=a.device)
        result[selection] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = {axis % a.ndim for axis in self.axes if -a.ndim <= axis < a.ndim}
        step = self.dilation + 1
        if step < 1:
            raise ValueError("dilation must be nonnegative")
        if step == 1 or not axes:
            return a
        selection = tuple(slice(None, None, step if axis in axes else 1)
                          for axis in range(a.ndim))
        return a[selection]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Dilate(self.axes, self.dilation, node.inputs[0].shape)(out_grad)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        stride, padding = self.stride, self.padding
        if stride < 1 or padding < 0:
            raise ValueError("convolution requires positive stride and nonnegative padding")
        if padding:
            A = A.pad(((0, 0), (padding, padding), (padding, padding), (0, 0)))

        batch, height, width, channels = A.shape
        kernel_h, kernel_w, kernel_channels, output_channels = B.shape
        if channels != kernel_channels:
            raise ValueError("input and kernel channel counts must match")
        if not (0 < kernel_h <= height and 0 < kernel_w <= width):
            raise ValueError("kernel must fit inside the padded input")
        out_h = (height - kernel_h) // stride + 1
        out_w = (width - kernel_w) // stride + 1

        n_stride, h_stride, w_stride, c_stride = A.strides
        windows = A.as_strided(
            (batch, out_h, out_w, kernel_h, kernel_w, channels),
            (n_stride, h_stride * stride, w_stride * stride,
             h_stride, w_stride, c_stride),
        )
        patch_size = kernel_h * kernel_w * channels
        patches = windows.compact().reshape((batch * out_h * out_w, patch_size))
        weights = B.compact().reshape((patch_size, output_channels))
        return (patches @ weights).reshape((batch, out_h, out_w, output_channels))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_tensor, weight_tensor = node.inputs
        x = input_tensor.realize_cached_data()
        weights = weight_tensor.realize_cached_data()
        grad = out_grad.realize_cached_data()
        padding = self.padding
        kernel_h, kernel_w = weights.shape[:2]
        batch, height, width, _ = x.shape
        if padding:
            x = x.pad(((0, 0), (padding, padding), (padding, padding), (0, 0)))

        # Recreate the stride-one output grid, inserting zeros at skipped windows.
        dense_shape = (batch, x.shape[1] - kernel_h + 1, x.shape[2] - kernel_w + 1,
                       grad.shape[3])
        dense_grad = Dilate((1, 2), self.stride - 1, dense_shape).compute(grad)
        unit_conv = Conv()

        # A full convolution with flipped kernels gives the padded input gradient.
        full_grad = dense_grad.pad(
            ((0, 0), (kernel_h - 1, kernel_h - 1), (kernel_w - 1, kernel_w - 1), (0, 0))
        )
        flipped_weights = weights.flip((0, 1)).permute((0, 1, 3, 2))
        input_grad = unit_conv.compute(full_grad, flipped_weights)
        input_grad = input_grad[:, padding:padding + height, padding:padding + width, :]

        # Use batch as the input channel so convolution sums over examples.
        weight_grad = unit_conv.compute(
            x.permute((3, 1, 2, 0)),
            dense_grad.permute((1, 2, 0, 3)),
        ).permute((1, 2, 0, 3))
        return (
            Tensor.make_const(input_grad, requires_grad=False),
            Tensor.make_const(weight_grad, requires_grad=False),
        )
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=0):
    return Conv(stride, padding)(a, b)


