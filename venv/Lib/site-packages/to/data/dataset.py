import time
from torch.utils.data import TensorDataset
from .reader import *


class DataSet(TensorDataset):

    def __init__(self, cfg, data_type, debug=True):
        self.cfg, self.data_type = cfg, data_type

        reader = DataReader(self.cfg, debug)

        if debug:
            p('Loading raw dataset "{}".'.format(data_type_name(data_type)))

        t0 = time.time()
        self.data, self.labels = reader[data_type]

        if debug:
            p('Done loading raw data in {:.3} secs.'.format(time.time() - t0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = [-1]
        if self.labels is not None:
            label = self.labels[i]
            if not hasattr(label, '__iter__'):
                label = [label]
        return to_tensor(np.array(self.data[i])), to_tensor(np.array(label))
