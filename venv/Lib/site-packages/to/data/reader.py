import numpy as np
import os
from ..utils.options import *
from ..utils.helpers import *

DEV, TRAIN, TEST = range(3)


class DataReader():

    def __init__(self, cfg, debug=True):
        self.cfg = cfg
        self.dev_set = None
        self.train_set = None
        self.test_set = None
        self.debug = debug
        default_directory = os.path.join(csd(), 'data')
        self.directory = get(self.cfg, DataOptions.Data_FOLDER.value, default=default_directory)

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = self.load_raw(self.directory, DEV)
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = self.load_raw(self.directory, TRAIN)
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = self.load_raw(self.directory, TEST)
        return self.test_set

    def __value_name(self, is_label):
        return 'labels' if is_label else 'features'

    def __get_path(self, data_type, is_label):
        options = [
            (DataOptions.DEV_DATA_FILE, DataOptions.DEV_LABELS_FILE), \
            (DataOptions.TRAIN_DATA_FILE, DataOptions.TRAIN_LABELS_FILE), \
            (DataOptions.TEST_DATA_FILE, None)
        ]
        key = options[data_type][is_label]

        if key is not None and has(self.cfg, key.value):
            return os.path.join(self.directory, get(self.cfg, key.value))
        else:
            return os.path.join(
                self.directory, '{}-{}.npy'.format(data_type_name(data_type), self.__value_name(is_label))
            )

    def __load_path(self, data_type, is_label):
        path = self.__get_path(data_type, is_label)
        if os.path.isfile(path):
            results = np.load(path, encoding='bytes')
            if self.debug:
                p('Dataset "{}" has {} records.'.format(path, len(results)))
            return results
        elif data_type == TEST and is_label:
            return None
        else:
            raise FileNotFoundError('File "{}" not found.'.format(path))

    def __save_path(self, data, data_type, is_label):
        path = self.__get_path(data_type, is_label)
        np.save(path, data, allow_pickle=False)

    def load_raw(self, path, data_type):
        return (self.__load_path(data_type, False), self.__load_path(data_type, True))

    def __len__(self):
        return 3

    def __getitem__(self, data_type):
        if data_type is DEV:
            return self.dev
        elif data_type is TRAIN:
            return self.train
        else:
            return self.test
