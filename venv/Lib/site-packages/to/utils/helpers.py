import os
import re
import csv
import sys
import glob
import torch
import inspect
import pathlib
import collections
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from enum import IntEnum
from inspect import signature
from .options import *
from .init import *

#----------------------------------------------------------------------------------------------------------
# Decorators
#----------------------------------------------------------------------------------------------------------


def static(varname, value):

    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


#----------------------------------------------------------------------------------------------------------
# Process module
#----------------------------------------------------------------------------------------------------------


def forward(module, args, kwargs):
    assert (isinstance(module, torch.nn.Module))
    args, kwargs = filter_args(module.forward, args, kwargs, module)
    return module(*args, **kwargs)


def filter_args(fn, args, kwargs, name=None):
    sig = signature(fn)
    pos, key = [], {}

    if name is None:
        name = fn.__name__

    for k, v in sig.parameters.items():
        if v.kind == ParameterKind.POSITIONAL_ONLY or \
            (v.kind == ParameterKind.POSITIONAL_OR_KEYWORD and v.default == v.empty):
            # Positional Parameter
            if len(args) == 0:
                raise Exception('Missing argument {} for "{}"'.format(v.name, name))
            pos.append(args.pop(0))
        elif v.kind == ParameterKind.KEYWORD_ONLY or \
            (v.kind == ParameterKind.POSITIONAL_OR_KEYWORD and v.default != v.empty):
            # Keyword Parameter
            if v.name not in kwargs:
                pos.append(v.default)
            else:
                pos.append(kwargs[v.name])
                del kwargs[v.name]
        elif v.kind == ParameterKind.VAR_POSITIONAL:
            # Variable Positional Parameter, a.k.a *args
            pos += args
        elif v.kind == ParameterKind.VAR_KEYWORD:
            # Variable Keyword Parameter, a.k.a **kwargs
            key = kwargs

    return pos, key


class ParameterKind(IntEnum):
    POSITIONAL_ONLY = 0
    POSITIONAL_OR_KEYWORD = 1
    VAR_POSITIONAL = 2
    KEYWORD_ONLY = 3
    VAR_KEYWORD = 4


#----------------------------------------------------------------------------------------------------------
# Sort and ArgSort
#----------------------------------------------------------------------------------------------------------


def argsort(lst, by=None, reverse=False):
    if by is not None:
        return sorted(range(len(lst)), key=lambda i: by(lst[i]), reverse=reverse)
    return sorted(range(len(lst)), key=lst.__getitem__, reverse=reverse)


def sort(lst, indexes=None, reverse=False):
    if indexes is not None:
        result = [lst[i] for i in indexes]
        return result[::-1] if reverse else result
    return sorted(lst)


#----------------------------------------------------------------------------------------------------------
# List Helpers
#----------------------------------------------------------------------------------------------------------


def __first(lmd, lst, asc=True):
    indexes = range(len(lst)) if asc else range(len(lst) - 1, -1, -1)
    for i in indexes:
        o = lst[i]
        if lmd(i, o, lst) == True:
            return i
    return -1


def first(lmd, lst):
    return __first(lmd, lst, True)


def last(lmd, lst):
    return __first(lmd, lst, False)


def where(lmd, lst, asc=True):
    indexes = range(len(lst)) if asc else range(len(lst) - 1, -1, -1)
    results = []
    for i in indexes:
        o = lst[i]
        if lmd(i, o, lst) == True:
            results.append(i)
    return results


#----------------------------------------------------------------------------------------------------------
# IO Helpers
#----------------------------------------------------------------------------------------------------------


def csd():
    if sys.argv[0] == '':
        return cwd()
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def cwd():
    return os.path.abspath(os.getcwd())


def filename(path):
    return os.path.basename(path)


def find_pattern(pattern, relative_to=None):
    files = glob.iglob(pattern, recursive=True)
    if relative_to is not None:
        prefix = csd()
        files = list(map(lambda x: os.path.relpath(x, prefix), files))
    return files


def match_prefix(word=None, suffix='.py', folder='configurations/'):
    options, patterns, folder = [], None, csd()

    if word == '' or word is None:
        patterns = ['{}**/*{}'.format(folder, suffix)]
    else:
        patterns = ['{}{}**/*{}'.format(folder, word, suffix), '{}{}*{}'.format(folder, word, suffix)]

    for pattern in patterns:
        pattern = os.path.join(folder, pattern)
        for path in find_pattern(pattern, relative_to=folder):
            path = path.replace(folder, '')
            options.append(path)

    return options


def touch(path, relative_to=csd()):
    pathlib.Path(os.path.join(relative_to, path)).touch()


def mkdirp(path, relative_to=csd()):
    return os.makedirs(os.path.join(relative_to, path), exist_ok=True)


def read_from_csv(path, as_type=int):
    with open(path, 'r') as csv_file:
        field_names = ['id', 'label']

        count, data = 0, []
        reader = csv.DictReader(csv_file, fieldnames=field_names)
        for row in reader:
            try:
                data.append(as_type(row['label']))
                count += 1
            except Exception as e:
                pass

        return count, data


def write_to_csv(data, path='submission.csv', field_names=['id', 'label']):
    with open(path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()

        for i, label in enumerate(data):
            writer.writerow({field_names[0]: str(i), field_names[1]: str(label)})


#----------------------------------------------------------------------------------------------------------
# Types
#----------------------------------------------------------------------------------------------------------


def is_int(i):
    return isinstance(i, int)


def is_float(i):
    return isinstance(i, float)


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
    else:
        return None


def is_num(i):
    return num(i) != None


def is_str(i):
    return isinstance(i, str)


def is_iter(i):
    return isinstance(i, collections.Iterable)


def is_dict(i):
    return isinstance(i, dict)


#----------------------------------------------------------------------------------------------------------
# Replace
#----------------------------------------------------------------------------------------------------------


def lreplace(s, old, new, count):
    return s.replace(old, new, count)


def rreplace(s, old, new, count):
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]


#----------------------------------------------------------------------------------------------------------
# Print and Log
#----------------------------------------------------------------------------------------------------------


def w(text=''):
    sys.stdout.write('{}'.format(text))
    sys.stdout.flush()


def p(text='', debug=True):
    if debug:
        print(
            '{} - {}() #{}: {}'.format(
                os.path.relpath(inspect.stack()[1][1], os.getcwd()),
                inspect.stack()[1][3],
                inspect.stack()[1][2], text
            )
        )
    else:
        print('{}'.format(text))


def ww(o, tab='    ', prefix=''):
    s = '{}'.format(o)
    s = prefix + _pprint(s, tab, '').replace('\n', '\n' + prefix)
    w(s)


def pp(o, tab='    ', prefix='', debug=False):
    s = '{}'.format(o)
    s = prefix + _pprint(s, tab, '').replace('\n', '\n' + prefix)
    p(s, debug=debug)


def ff(s, tab='    ', prefix=''):
    if hasattr(s, '__repr__'):
        s = repr(s)
    else:
        s = '{}'.format(s)
    s = re.sub(', *', ',', s)
    r = []

    def is_named_arg(idx):
        return beyond(s, idx, lambda i, c, s: c == '=', lambda i, c, s: c in [',', ')'])

    def is_class(idx):
        return beyond(s, idx, lambda i, c, s: c.isupper(), lambda i, c, s: c in [',', ')'])

    for i, c in enumerate(s):
        r.append(c)
        if c == '[' or c == '{':
            prefix = prefix + tab
            r.append('\n')
            r.append(prefix)
        elif c == '(':
            prefix = prefix + tab
            if is_class(i):
                r.append('\n')
                r.append(prefix)
        elif c == ']' or c == '}':
            prefix = rreplace(prefix, tab, '', 1)
            r.pop(-1)
            r.append('\n')
            r.append(prefix)
            r.append(c)
        elif c == ')':
            prefix = rreplace(prefix, tab, '', 1)
        elif c == ',':
            if is_named_arg(i) or is_class(i):
                r.append('\n')
                r.append(prefix)
    return ''.join(r)


#----------------------------------------------------------------------------------------------------------
# Look ahead & Look beyond
#----------------------------------------------------------------------------------------------------------


def ahead(A, i, lmd, brk):
    for j in range(0, i):
        o = A[j]
        if brk(j, o, A):
            return False
        elif lmd(j, o, A):
            return True


def beyond(A, i, lmd, brk):
    for j in range(i + 1, len(A)):
        o = A[j]
        if brk(j, o, A):
            return False
        elif lmd(j, o, A):
            return True


#----------------------------------------------------------------------------------------------------------
# Get & Has
#----------------------------------------------------------------------------------------------------------
def has_index(l, i):
    return l is not None and i >= 0 and i < len(l)


def has(o, *k):
    if len(k) == 1:
        if is_int(k[0]) and has_index(o, k[0]):
            return True
        elif is_str(k[0]) and hasattr(o, k[0]):
            return True
        elif is_dict(o) and k[0] in o:
            return True
    elif len(k) > 1:
        if is_int(k[0]) and has_index(o, k[0]):
            return has(o[k[0]], *k[1:])
        elif is_str(k[0]) and hasattr(o, k[0]):
            return has(get(o, k[0]), *k[1:])
        elif is_dict(o) and k[0] in o:
            return has(o[k[0]], *k[1:])
    return False


def get(o, *k, default=None):
    if len(k) == 0:
        return default
    elif len(k) == 1:
        if is_int(k[0]) and has_index(o, k[0]):
            return o[k[0]]
        elif is_str(k[0]) and hasattr(o, k[0]):
            return getattr(o, k[0])
        elif is_dict(o) and k[0] in o:
            return o[k[0]]
        else:
            return default
    else:
        if is_int(k[0]) and has_index(o, k[0]):
            return get(o[k[0]], *k[1:], default=default)
        elif is_str(k[0]) and hasattr(o, k[0]):
            return get(getattr(o, k[0]), *k[1:], default=default)
        elif is_dict(o) and k[0] in o:
            return get(o[k[0]], *k[1:], default=default)
        else:
            return default


def inheritance(cls):
    if not inspect.isclass(cls):
        cls = cls.__class__
    return list(inspect.getmro(cls))


#----------------------------------------------------------------------------------------------------------
# PyTorch helpers
#----------------------------------------------------------------------------------------------------------


def repackage_hidden(h):
    if type(h) == torch.autograd.Variable:
        return torch.autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def to_tensor(array):
    if isinstance(array, list):
        array = np.array(array)
    return torch.from_numpy(array).float()


def to_variable(tensor):
    if torch.cuda.is_available() and hasattr(tensor, 'cuda'):
        tensor = tensor.cuda()

    if not torch.is_tensor(tensor):
        return tensor
    return torch.autograd.Variable(tensor)


def init_model_parameters(module):
    init_options = get(module.cfg, InitOptions.INIT_OPTIONS.value)
    if init_options is None:
        init_options = {
            'Conv': {
                'weight': Init.xavier_uniform(),
                'bias': Init.uniform(),
            },
            nn.Linear: {
                'weight': Init.xavier_uniform(),
                'bias': Init.uniform(),
            },
            'RNNBase': {
                'weight': Init.orthogonal(),
                'bias': Init.uniform(),
            },
        }

    for m in module.modules():
        for k, v in init_options.items():
            should_init = False
            if is_str(k):
                classes = inheritance(m)
                should_init = any([k in c.__name__ for c in classes])
            elif isinstance(k, nn.Module) and isinstance(m, k):
                should_init = True

            if should_init:
                map(lambda x: init_layer_parameters(m, x[0], x[1]), list(zip(v.keys(), v.values())))


def init_layer_parameters(m, key, fn):
    for name, param in m.named_parameters():
        if key in name:
            fn.apply(param)


def collate(data, axis=1, dim=2, mode='constant', value=0, min_len=None, concat_labels=False):
    axis, dim = axis - 1, dim - 1
    # axis and dim are on a per row basis
    data.sort(key=lambda x: len(x[0]), reverse=True)

    results = list(zip(*data))
    data, labels = results[0], results[1]

    lengths = [row.shape[axis] for row in data]
    max_len = max(lengths)

    if min_len is not None and max_len < min_len:
        max_len = min_len

    pad_locs = [(0, 0) * (dim - axis - 1) + (0, max_len - row.shape[axis]) for i, row in enumerate(data)]

    results[0] = torch.stack([F.pad(row, pad_locs[i], mode, value) for i, row in enumerate(data)])

    if concat_labels:
        results[1] = torch.cat(labels)
    else:
        results[1] = torch.stack(labels)
    one_hot = to_tensor(np.array([[1] * length + [0] * (max_len - length) for length in lengths]))

    return tuple(results + [lengths, one_hot])


def mask(data, one_hot, axis=1):
    if isinstance(data, torch.autograd.Variable):
        data = data.data.cpu().numpy()
    elif isinstance(data, torch.tensor._TensorBase):
        data = data.cpu().numpy()

    results = []
    for i in range(len(data)):
        assert data[i].shape[axis - 1] == one_hot[i].shape[0]
        length = int(one_hot[i].sum())
        results.append(data[i].take(list(range(0, length)), axis=axis - 1))
    return np.array(results, dtype='object')
    # print(data[1].shape)
    # print(one_hot[1].shape)
    # print('///////////////////')
    # print(one_hot[1].cpu().numpy().sum())
