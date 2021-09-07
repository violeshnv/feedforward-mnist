#!/usr/bin/env python
# -*- coding:utf-8 -*-

from random import choices

from data_loader import *
from processor import *
from backward_processor import *


def get_network(network_path):
    npys = list(int(n[0:-4])
                for n in listdir(network_path) if n.endswith(".npy"))
    count = max(npys) if npys else None
    if count:
        return count, np.load(f"{network_path}\\{count}.npy")
    else:
        return 0, None


def main():
    backward()


def forward():
    bound = 80
    step_length = 0.05
    times = 20

    data = DataLoader("resource").resource
    count, network = get_network("networks")
    print(count, network)

    proc = Processor((784, 28, 16, 10), data, network)
    proc.load("train-images", "train-labels")

    while True:
        print(f"ENTER [{count}:{count+bound}]")
        proc.input_choices(choices(range(60000), k=bound))
        proc.multi_gradient_descent(times, step_length)

        Recorder.record(proc.network, count, "networks\\")

        print(f"EXIT  [{count}:{count+bound}]")
        count += bound


def backward():
    bound = 200
    step_length = 0.005
    times = 20

    data = DataLoader("resource").resource
    count, network = get_network("backward_network")
    print(count, network)

    proc = BackwardProcessor((784, 64, 16, 10), data, network)
    proc.load("train-images", "train-labels")

    while True:
        print(f"ENTER [{count}:{count+bound}]")
        proc.input_choices(choices(range(60000), k=bound))
        proc.multi_gradient_descent(times, step_length)

        if count % (1000 * bound) == 0:
            Recorder.record(proc.network, count, "backward_network\\")

        print(f"EXIT  [{count}:{count+bound}]")
        count += bound


if __name__ == "__main__":
    main()
