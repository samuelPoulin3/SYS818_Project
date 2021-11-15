#! /usr/bin/env python3
from plugins import layers
from plugins.ScanShow.ScanShow import ScanShow
DEBUG = False

class Models():
    """
        The model of our algorithm     
    """
    __slots__ = ('steps', 'new_mat')
    def __init__(self, *args, **kwargs):
        self.steps = []
        self.new_mat = {} 

    def add(self, layers):
        self.steps.append(layers)

    def compile(self):
        pass

    def fit(self, data, labels, epochs, batch_size):
        dispatch = {'Registration': layers.Registration_compile,
                    'Gauss2D': layers.Gauss2D_compile,
                    'FeatExtract': layers.FeatExtract_compile,
                    'Conv2D': layers.Conv2D_compile,
                    'MaxPooling2D': layers.MaxPooling2D_compile}

        data_by_epoch = data
        for step in self.steps:
            if self.new_mat == {}:
                self.new_mat[step['name']] = dispatch[step['name']](data_by_epoch, step)

            else:
                self.new_mat[step['name']] = dispatch[step['name']](self.new_mat[previous_step], step)
            previous_step = step['name']
