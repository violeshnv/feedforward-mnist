#!/usr/bin/env python
# -*- coding:utf-8 -*-

from abc import abstractmethod

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    y = sigmoid(x)
    return (1 - y) * y


def cross_entropy_error(result, labels):
    return -np.sum(np.log(result[np.arange(labels.size), labels] + 1e-7))


def soft_max(x):
    t = np.exp(x - x.max())
    y = np.sum(t, axis=1).reshape(t.shape[0], 1)
    return t / y


def square_loss(result, labels):
    t = np.array(result)
    t[np.arange(labels.size), labels] -= 1
    return np.sum(t ** 2)


def relu(x):
    return np.maximum(x, 0)


class Layer:
    @abstractmethod
    def forward(self, *args):
        """"""
    @abstractmethod
    def backward(self, dout):
        """"""


class AddLayer(Layer):
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class MulLayer(Layer):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class SigmoidLayer(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 - np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class SoftmaxWithLossLayer(Layer):
    def __init__(self):
        self.loss = None
        self.t = None
        self.y = None

    def forward(self, x, t):
        self.t = t
        self.y = soft_max(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout):
        size = self.t.shape[0]
        return (self.y - self.t) / size


class ReluLayer(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dout):
        return np.where(self.x > 0, 1, 0) * dout
