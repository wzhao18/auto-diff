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


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if not param.grad:
                continue
            if param not in self.u:
                self.u[param] = 0.
    
            grad = param.grad.data + param.data * self.weight_decay
            grad_w_momentum = self.momentum * self.u[param] + (1. - self.momentum) * grad
            self.u[param] = grad_w_momentum

            param.data = param.data - self.lr * grad_w_momentum
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
        self.t += 1
        for param in self.params:
            if not param.grad:
                continue
            if param not in self.m:
                self.m[param] = 0.
            if param not in self.v:
                self.v[param] = 0.

            grad = param.grad.data + param.data * self.weight_decay
            m = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            v = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            m_bias_corr = m / (1 - self.beta1 ** self.t)
            v_bias_corr = v / (1 - self.beta2 ** self.t)

            self.m[param] = m
            self.v[param] = v

            param.data = param.data - self.lr * (m_bias_corr / (v_bias_corr ** 0.5 + self.eps))
        ### END YOUR SOLUTION
