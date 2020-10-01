import torch
from torch import nn
from enum import IntEnum
from ..utils.helpers import *
from ..utils.options import *


class Mode(IntEnum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class NeuralNetwork(torch.nn.Module):

    def __init__(self, cfg):
        super(NeuralNetwork, self).__init__()
        self.cfg = cfg

        layers = get(self.cfg, NeuralNetworkOptions.LAYERS.value, default=[])
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '()'
