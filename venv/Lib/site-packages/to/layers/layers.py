import torch
import torch.nn as nn
import math
from ..utils.helpers import *


class BaseLayer(torch.nn.Module):

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TransposeLayer(BaseLayer):

    def __init__(self, d1=0, d2=1):
        super(TransposeLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2

    def forward(self, h):
        return h.transpose(self.d1, self.d2)

    def __repr__(self):
        return self.__class__.__name__ + '({}, {}})'.format(self.d1, self.d2)


class PermuteLayer(BaseLayer):

    def __init__(self, *args):
        super(PermuteLayer, self).__init__()
        self.order = args

    def forward(self, h):
        return h.permute(*self.order)

    def __repr__(self):
        return self.__class__.__name__ + '{}'.format(tuple(self.order)).replace(',)', ')')


class LogShapeLayer(BaseLayer):

    def forward(self, h):
        p(h.shape)
        return h


class PrintLayer(BaseLayer):

    def forward(self, h):
        print()
        return h


class PackPaddedLayer(BaseLayer):

    def __init__(self, batch_first=False):
        super(PackPaddedLayer, self).__init__()
        self.batch_first = batch_first

    def forward(self, h, lengths):
        h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=self.batch_first)
        return h


class PadPackedLayer(BaseLayer):

    def __init__(self, batch_first=False):
        super(PadPackedLayer, self).__init__()
        self.batch_first = batch_first

    def forward(self, h):
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
        return h


class Flatten(BaseLayer):

    def forward(self, input):
        return input.view(input.size(0), -1)


class MeanPoolingLayer(BaseLayer):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, input, dim=2):
        length = input.shape[2]
        return torch.sum(input, dim=2) / length
