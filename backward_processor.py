#!/usr/bin/env python
# -*- coding:utf-8 -*-

from operator import mul

from recorder import *
from layer import *

from typing import Union, List, NoReturn


class BackwardProcessor:
    """
    反向传播NN
    """

    def __init__(self, node_nums: tuple, data: dict, network: np.array = None):
        """
        :param node_nums: 一个包含每一层的节点数的元组
                          a tuple containing the number of nodes at each layer
        :param data: 可以在load()中根据键索引数据，且数据为array类型，按行读取
                     the data should be able to be indexed in load(key)
        :param network: 已存在的神经网络数据 neural network data
        """
        self.node_nums = node_nums
        self.data = data

        self.shapes = tuple(zip(self.node_nums, self.node_nums[1:]))
        self.lengths = tuple(mul(*shape) for shape in self.shapes)

        self.offset = sum(self.lengths)
        self.total_length = self.offset + sum(self.node_nums[1:])

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
                        else np.random.normal(size=self.total_length))

        self.grad = np.empty_like(self.network)

        # to be loaded

        self.images = None
        self.labels = None

        self.input_images = None
        self.input_labels = None

        self.layer = None
        self.layers = None

        self.node_errors = None

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

    def process(self, network: np.ndarray = None) -> np.array:
        """
        :param network: 使用的神经网络，默认类中已加载的的神经网络
        :return:
        """
        network = network or self.network
        self.layers = []
        self.layer = self.input_images
        self.layers.append(self.layer)
        for i in range(self.ls - 1):
            self.layer = sigmoid(self.layer)
            self.layer = (self.layer @
                          network[self.ws[i]].reshape(self.shapes[i]) +
                          network[self.bs[i]])
            self.layers.append(self.layer)

        return self.layer

    def predict(self, network: np.ndarray = None) -> np.array:
        """
        use softmax to standardize the probabilities
        :param network:
        :return:
        """
        return soft_max(self.process(network))

    def loss(self, x: np.ndarray) -> np.ndarray:
        """
        使用交叉熵求损失
        using CrossEntropy to calculate the loss
        :param x: 预测标签
        :return: 损失值
        """
        return cross_entropy_error(x, self.input_labels)

    def output_layer_gradient(self):
        size = self.input_labels.shape[0]
        t = np.zeros((size, 10))
        t[np.arange(size), self.input_labels] = 1
        return self.predict() - t

    def set_node_errors(self) -> NoReturn:
        """
        calculate node error for each node
        """
        self.node_errors = list(range(self.ls - 1))
        self.node_errors[-1] = self.output_layer_gradient()
        for i in range(self.ls - 3, -1, -1):
            self.node_errors[i] = ((
                self.node_errors[i + 1] @
                self.network[self.ws[i + 1]].reshape(self.shapes[i + 1]).T) *
                derivative_sigmoid(self.layers[i + 1]))

    def numerical_gradient(self) -> np.ndarray:
        """
        calculate gradient for each node using
        """
        self.set_node_errors()
        size = self.input_labels.shape[0]
        for i in range(self.ls - 2, -1, -1):
            self.grad[self.ws[i]] = (sigmoid(self.layers[i].T) @
                                     self.node_errors[i]).flatten() / size
            self.grad[self.bs[i]] = self.node_errors[i].sum(axis=0) / size
        return self.grad

    def gradient_descent(self, step_length: float = 0.01) -> NoReturn:
        """
        use gradient descent to get new network
        :param step_length: learning rate
        """
        self.network -= self.numerical_gradient() * step_length

    def multi_gradient_descent(self,
                               times: int = 20,
                               step_length: float = 0.01) -> NoReturn:
        """
        call gradient_descent() several times and output
        :param times: loop for how many times
        :param step_length: learning rate
        """
        for _ in range(times):
            self.gradient_descent(step_length)
        Recorder.output(self.predict().argmax(axis=1) == self.input_labels,
                        self.loss(self.predict()),
                        np.sum(self.grad ** 2))


def predict(proc: BackwardProcessor, images: np.ndarray) -> np.ndarray:
    """
    :param proc: BackwardProcessor
    :param images: images to be predicted
    :return: predicted value
    """
    assert images.shape[1] == 784, f"images shape unmatched: {images.shape}"
    proc.input_images = images
    return proc.predict().argmax(axis=1)
