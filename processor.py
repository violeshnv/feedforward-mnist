#!/usr/bin/env python
# -*- coding:utf-8 -*-

from operator import mul

from layer import *
from recorder import *

from typing import Union, List, NoReturn


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def cross_entropy_error(result: np.ndarray, labels: np.ndarray):
    return -(np.sum(np.log(result[np.arange(labels.size), labels] + 1e-5)))


def soft_max(x: np.ndarray):
    t = np.exp(x - np.max(x))
    y = np.sum(t, axis=1).reshape(t.shape[0], 1)
    return t / y


class Processor:
    """
    正向传播NN
    """

    def __init__(self, node_nums: tuple, data: dict, network: np.array = None):
        """
        :param node_nums: 一个包含每一层的节点数的元组
                          a tuple containing the number of nodes at each layer
        :param data: 可以在load()中根据键索引数据，且数据为array类型，按行读取
                     the data should be able to be indexed in load(key)
        :param network: 已存在的神经网络数据 neural network data
        """
        self.node_nums = node_nums  # 每层节点数
        self.data = data  # 数据

        self.shapes = tuple(zip(self.node_nums, self.node_nums[1:]))  # 每层矩阵形状
        self.lengths = tuple(mul(*shape) for shape in self.shapes)  # 每层矩阵大小

        self.offset = sum(self.lengths)  # 划分权重和偏置
        self.total_length = self.offset + sum(self.node_nums[1:])  # 神经网络长度

        self.ls = len(self.node_nums)
        self.ws = tuple(
            slice(sum(self.lengths[0:i]), sum(self.lengths[0:i + 1]))
            for i in range(self.ls - 1))
        self.bs = tuple(
            slice(self.offset + sum(self.node_nums[1:i]),
                  self.offset + sum(self.node_nums[1:i + 1]))
            for i in range(1, self.ls))

        assert network is None or network.size == self.total_length
        self.network = (network if network is not None
                        else np.random.randn(self.total_length))

        self.grad = np.empty_like(self.network)

        # to be loaded

        self.images = None
        self.labels = None

        self.input_images = None
        self.input_labels = None

        self.layer = None

    def load(self, images: str, labels: str) -> NoReturn:
        """
        从data加载数据
        load data by file name without suffix
        :param images: 训练图像 training images
        :param labels: 训练标签 training labels
        """
        self.images = self.data[images] / 255  # standardize
        self.labels = self.data[labels]

    def input(self, start: int, stop: int) -> NoReturn:
        """
        一次加载几个数据，batch
        input interval to choose data, [start, stop)
        :param start: 区间开始点，闭
        :param stop: 区间结束点，开
        """
        self.input_images = self.images[start:stop]
        self.input_labels = self.labels[start:stop]

    def input_choices(self, choices: Union[List, np.ndarray]) -> NoReturn:
        """
        一次加载几个数据，batch
        input list to choose data
        :param choices: 用于array的列表索引，比如[1,7,8]选择第1,7,8个数据
        """
        self.input_images = self.images[choices]
        self.input_labels = self.labels[choices]

    def predict(self, network: np.ndarray = None) -> np.ndarray:
        """
        使用softmax获得预测结果的概率
        obtain the probabilities of results (namely 0-9) using softmax
        :param network: 使用的神经网络，默认类中已加载的的神经网络
        :return: 预测结果概率
        """
        network = network or self.network
        self.layer = self.input_images
        for i in range(3):
            self.layer = sigmoid(
                self.layer @ network[self.ws[i]].reshape(self.shapes[i]) +
                network[self.bs[i]])
        self.layer = soft_max(self.layer)
        return self.layer

    def loss(self, x: np.ndarray) -> np.ndarray:
        """
        使用交叉熵求损失
        using CrossEntropy to calculate the loss
        :param x: 预测标签
        :return: 损失值
        """
        return cross_entropy_error(x, self.input_labels)

    def numerical_gradient(self, step_length: float = 0.01) -> NoReturn:
        """
        关键函数，正向求梯度
        key function: forward gradient
        :param step_length: 训练步长，学习率 learning rate
        """
        h = 1e-5
        for i in range(self.grad.size):
            t = self.network[i]

            self.network[i] = t - h
            f1 = self.loss(self.predict())
            self.network[i] = t + h
            f2 = self.loss(self.predict())

            self.grad[i] = (f1 - f2) / (2 * h)

            self.network[i] = t

        self.grad *= step_length

    def gradient_descent(self, step_length: float = 0.01) -> NoReturn:
        """
        求梯度，梯度下降，使用Recorder.output输出
        find the gradient; gradient descent; using the Recorder.output to output
        :param step_length: 训练步长，学习率 learning rate
        """
        self.numerical_gradient(step_length)
        self.network += self.grad
        Recorder.output(self.predict().argmax(axis=1) == self.input_labels,
                        self.loss(self.layer),
                        np.sum(self.grad ** 2))

    def multi_gradient_descent(
            self,
            times: int = 20,
            step_length: float = 0.01):
        for _ in range(times):
            self.gradient_descent(step_length)
