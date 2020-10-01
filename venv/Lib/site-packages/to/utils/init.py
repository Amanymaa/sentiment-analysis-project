import torch.nn as nn


class Init():

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def apply(self, m):
        self.fn(m, *self.args, **self.kwargs)

    def uniform(*args, **kwargs):
        return Init(nn.init.uniform, *args, **kwargs)

    def normal(*args, **kwargs):
        return Init(nn.init.normal, *args, **kwargs)

    def constant(val):
        return Init(nn.init.constant, val)

    def eye(*args, **kwargs):
        return Init(nn.init.eye, *args, **kwargs)

    def dirac(*args, **kwargs):
        return Init(nn.init.dirac, *args, **kwargs)

    def xavier_uniform(*args, **kwargs):
        return Init(nn.init.xavier_uniform, *args, **kwargs)

    def xavier_normal(*args, **kwargs):
        return Init(nn.init.xavier_normal, *args, **kwargs)

    def kaiming_uniform(*args, **kwargs):
        return Init(nn.init.kaiming_uniform, *args, **kwargs)

    def kaiming_normal(*args, **kwargs):
        return Init(nn.init.kaiming_normal, *args, **kwargs)

    def orthogonal(*args, **kwargs):
        return Init(nn.init.orthogonal, *args, **kwargs)

    def sparse(sparsity, **kwargs):
        return Init(nn.init.sparse, sparsity, **kwargs)

    def zeros(*args, **kwargs):
        return Init(nn.init.constant, 0)

    def ones(*args, **kwargs):
        return Init(nn.init.constant, 1)
