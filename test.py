#!/usr/bin/env python
# -*- coding:utf-8 -*-

from main import *


def test_my_images(images: np.ndarray):
    """
    :param images: images drawn by myself
    """
    assert images.shape[1] == 784, images.shape

    _, network = get_network("backward_network")
    proc = BackwardProcessor((784, 64, 16, 10), {}, network)

    print(predict(proc, images / 255))


def test_default_images(row: int, column: int, start: int = 0):
    data_load = DataLoader("resource")
    data = data_load.resource
    count, network = get_network("backward_network")
    proc = BackwardProcessor((784, 64, 16, 10), data, network)

    proc.load("t10k-images", "t10k-labels")
    proc.input(start, start + row * column)
    print(proc.predict().argmax(axis=1).reshape(column, row).T)

    data_load.display_images(row, column)


def forward_test():
    data = DataLoader("resource").resource
    count, network = get_network("networks")
    proc = Processor((784, 64, 16, 10), data, network)

    print("t10k")
    proc.load("t10k-images", "t10k-labels")
    proc.input(0, 10000)
    Recorder.less_output(proc.predict().argmax(axis=1) == proc.input_labels,
                         proc.loss(proc.predict()))

    print("train")
    proc.load("train-images", "train-labels")
    proc.input(0, 60000)
    Recorder.less_output(proc.predict().argmax(axis=1) == proc.input_labels,
                         proc.loss(proc.predict()))


def backward_test():
    data = DataLoader("resource").resource
    count, network = get_network("backward_network")
    proc = BackwardProcessor((784, 64, 16, 10), data, network)

    print("t10k")
    proc.load("t10k-images", "t10k-labels")
    proc.input(0, 10000)
    Recorder.less_output(proc.predict().argmax(axis=1) == proc.input_labels,
                         proc.loss(proc.predict()))

    print("train")
    proc.load("train-images", "train-labels")
    proc.input(0, 60000)
    Recorder.less_output(proc.predict().argmax(axis=1) == proc.input_labels,
                         proc.loss(proc.predict()))
