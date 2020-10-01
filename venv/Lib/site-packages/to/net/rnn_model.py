import torch.nn as nn
from ..layers.rnn import is_rnn
from .base import *
from ..utils.helpers import *


class RNNModel(NeuralNetwork):

    def __init__(self, cfg):
        super(RNNModel, self).__init__(cfg)

        self.in_features = get(self.cfg, RNNModelOptions.IN_CHANNELS.value, default=40)
        self.out_features = get(self.cfg, RNNModelOptions.OUT_CHANNELS.value, default=47)

        start_lmd = lambda i, o, A: i >= 0 and not is_rnn(get(A, i - 1)) and is_rnn(o)
        end_lmd = lambda i, o, A: i <= len(A) - 1 and is_rnn(o) and not is_rnn(get(A, i + 1))
        self.starts = where(start_lmd, self.layers)
        self.ends = where(end_lmd, self.layers)

        self.should_pack_padded = get(self.cfg, RNNModelOptions.PACK_PADDED.value, default=False) and \
            len(self.starts) >= 0 and len(self.ends) >= 0

    def __forward(self, h, lengths=None):

        for i, layer in enumerate(self.layers):
            if is_rnn(layer):
                if self.should_pack_padded and i in self.starts:
                    h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=False)

                # if is_pytorch_rnn(layer):
                #     layer.flatten_parameters()
                h, _ = layer(h)

                if self.should_pack_padded and i in self.ends:
                    h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
            else:
                h = forward(layer, [h, lengths], {})

        return h

    def forward(self, inputs, lengths=None, forward=0, stochastic=False):
        h = inputs.float()  # (n, t)

        h = self.__forward(h, lengths)

        if stochastic:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel

        logits = h
        if forward > 0:
            outputs = []

            h = torch.max(logits[:, -1:, :], dim=2)[1] + 1
            for i in range(forward):
                h = self.__forward(h)

                if stochastic:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel

                outputs.append(h, lengths)

                h = torch.max(h, dim=2)[1] + 1
            logits = torch.cat([logits] + outputs, dim=1)

        return logits
