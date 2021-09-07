# !/usr/bin/env python
# -*- coding:utf-8 -*-

from PIL import Image
from os import listdir, path


import numpy as np

C = 28
S = 784


class DataLoader:

    def __init__(self, dir_path: str):
        """
        load data (images and labels)
        :param dir_path: directory that contains training files
        """
        self.resource = {}
        file_names = listdir(dir_path)
        for file_name in file_names:
            full_path = path.join(dir_path, file_name)
            name, _ = path.splitext(file_name)
            if "images" in name:
                images = np.fromfile(full_path, dtype=np.uint8, offset=16)
                self.resource[name] = images.reshape(images.size // S, S)
            elif "labels" in name:
                labels = np.fromfile(full_path, dtype=np.uint8, offset=8)
                self.resource[name] = labels

    def display_images(self, row: int, column: int,
                       interval: slice = slice(None, None, None)):
        background = Image.new("L", (column * C, row * C))
        resource = self.resource["t10k-images"][interval]
        for i in range(column):
            for j in range(row):
                index = i * row + j
                array = resource[index].reshape(C, C)
                img = Image.fromarray(array)
                background.paste(img, (i * 28, j * 28))
        background.show()

    def __getitem__(self, item: str):
        return self.resource[item]
