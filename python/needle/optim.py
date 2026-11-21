"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def clip_grad_norm(self, max_norm=0.25):
        """Scale all gradients together to respect a global L2 norm limit."""
        ### BEGIN YOUR SOLUTION
        if max_norm < 0:
            raise ValueError("max_norm must be nonnegative")
        parameters = [p for p in self.params if p.grad is not None]
        squared_norm = sum(
            np.square(p.grad.numpy(), dtype=np.float64).sum() for p in parameters
        )
        norm = np.sqrt(squared_norm)
        if norm > max_norm:
            scale = float(max_norm / norm)
            for param in parameters:
                param.grad = param.grad.detach() * scale
        ### END YOUR SOLUTION


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # get fisrt weight
        for param in self.params:
            if param not in self.u:
                self.u[param] = ndl.init.zeros(
                    *param.shape, device=param.device, dtype=param.dtype
                )
            gradient = param.grad.data + self.weight_decay * param.data
            new_u = self.momentum * self.u[param] + (1 - self.momentum) * gradient
            self.u[param] = new_u
            param.data = param.data - self.lr * new_u
        ### END YOUR SOLUTION

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t = self.t + 1
        for param in self.params:
            if param not in self.m:
                self.m[param] = ndl.init.zeros(
                    *param.shape, device=param.device, dtype=param.dtype
                )
            if param not in self.v:
                self.v[param] = ndl.init.zeros(
                    *param.shape, device=param.device, dtype=param.dtype
                )

            # calc gradient
            gradient = param.grad.data + self.weight_decay * param.data

            # new u and v
            new_u = self.beta1 * self.m[param] + (1 - self.beta1) * gradient
            new_v = self.beta2 * self.v[param] + (1 - self.beta2) * gradient**2

            # bias correction
            bias_u = new_u / (1 - self.beta1**self.t)
            bias_v = new_v / (1 - self.beta2**self.t)

            # update
            self.m[param] = new_u
            self.v[param] = new_v
            param.data = param.data - self.lr * (
                bias_u / (bias_v ** (1 / 2) + self.eps)
            )
        ### END YOUR SOLUTION
