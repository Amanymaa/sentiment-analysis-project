import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.rnn import PackedSequence


def RNN(
    mode,
    input_size,
    hidden_size,
    bias=True,
    dropout=0,
    num_layers=1,
    batch_first=False,
    bidirectional=False,
    zoneout=None,
    residual=False,
    weight_drop=0,
    weight_norm=False,
    attention_layer=None
):
    params = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional
    )

    modes = {
        'RNN': (nn.RNNCell, nn.RNN),
        'GRU': (nn.GRUCell, nn.GRU),
        'LSTM': (nn.LSTMCell, nn.LSTM),
    }

    if mode not in modes.keys():
        raise Exception('Unknown mode: {}'.format(mode))

    wn_func = wn if weight_norm else lambda x: x
    with_attention = attention_layer is not None
    has_zoneout = zoneout is not None

    if zoneout is not None or \
         residual is True or \
         weight_drop is not 0 or \
         weight_norm is True or \
         attention_layer is not None:
        raise NotImplementedError('Not yet implemented.')

    if has_zoneout:
        raise NotImplementedError('Zoneout is not yet implemented.')
    else:
        rnn = modes[mode][1]

        if residual:
            rnn = _wrap_stacked_recurrent(rnn, num_layers=num_layers, weight_norm=weight_norm, residual=True)
            params['num_layers'] = 1

        module = wn_func(rnn(**params))

    return module


def is_pytorch_rnn(x):
    return isinstance(x, nn.modules.rnn.RNNBase)


def is_rnn(x):
    return isinstance(x, nn.modules.rnn.RNNBase) or \
        isinstance(x, _BaseRNNModule)


#----------------------------------------------------------------------------------------------------------
# Base Classes
#----------------------------------------------------------------------------------------------------------


class _BaseRNNModule(nn.Module):

    def __repr__(self):
        return self.__class__.__name__ + '()'
