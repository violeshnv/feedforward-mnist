#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


class Recorder:

    @staticmethod
    def record(array: np.array, count: int, file_path: str):
        np.save(file_path + str(count), array)

    @staticmethod
    def output(comp, loss, grad):
        print(f"\t\taccuracy: {(np.sum(comp) / comp.size):.2%}")
        print(f"\t\tloss: {loss:18.10f}")
        print(f"\t\tgrad: {grad:18.10f}")

    @staticmethod
    def less_output(comp, loss=None):
        print(f"accuracy: {(np.sum(comp) / comp.size):.2%}")
        if loss:
            print(f"\t\tloss: {loss:18.10f}")
