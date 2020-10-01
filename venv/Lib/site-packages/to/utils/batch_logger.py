import numpy as np
import time
import re
from collections import defaultdict
from .options import *
from .helpers import *
from ..net.base import Mode


class Logger(object):

    def __init__(self, trainer):
        super(Logger, self).__init__()
        self.trainer = trainer

    def __reset(self, mode=Mode.TRAIN):
        self.print_interval = get(self.trainer.cfg, TrainerOptions.PRINT_INVERVAL.value, default=100)
        self.print_accuracy = get(self.trainer.cfg, TrainerOptions.PRINT_ACCURACY.value, default=True)
        self.batch_size = get(self.trainer.cfg, TrainerOptions.BATCH_SIZE.value, default=64)

        self.mode = mode
        self.t0 = time.time()
        self.__reset_epoch()

    def __reset_epoch(self):
        self.extra_logs = defaultdict(list)
        self.losses, self.batch_count = [], 0
        self.interval_correct, self.interval_count = 0, 0
        self.all_correct, self.all_count = 0, 0
        self.t1 = time.time()

    def start(self, mode=Mode.TRAIN):
        self.__reset(mode)

    def start_epoch(self):
        self.__reset_epoch()

    def increment(self):
        self.batch_count += 1

    def log_loss(self, loss, **extra_logs):
        self.losses.append(loss)
        for k, v in extra_logs.items():
            self.extra_logs[k].append(v)

    def log_batch(self, mode, x, y, extras, y_hat):
        if self.print_accuracy and self.mode is not Mode.TEST:
            match_results = self.trainer._match(mode, x, y, extras, y_hat)

            correct = match_results.sum()
            self.interval_correct += correct
            self.all_correct += correct
            self.interval_count += x.size
            self.all_count += x.size

    def print_summary(self):
        template, percentage = '', 0.0
        if self.print_accuracy:
            percentage = self.all_correct / max(self.all_count, 1) * 100
            if percentage == 0:
                percentage = None
            template += '{0} / {1} ({2:.2f} %) correct! '

        template += 'Loss is: {3:.6f}/{4:.6f}/{5:.6f}'

        i = max([int(i) for i in re.findall('(?<={)\d*(?=[}:])', template)])
        extras = []
        for k, v in self.extra_logs.items():
            i += 1
            template += ' | {}: {}{}{}'.format(k, '{', i, '}')
            extras.append(np.mean([np.mean(i) for i in v]))

        min_loss, mean_loss, total_loss = self.get_loss()
        p(template.format(self.all_correct, max(self.all_count, 1), percentage, min_loss, \
            mean_loss, total_loss, *extras))

    def print_batch(self, check_print_interval=True):
        if check_print_interval and self.batch_count % self.print_interval != 0:
            return

        curr_time = time.time()
        batch_time = curr_time - self.t1
        total_time = curr_time - self.t0
        self.t1 = time.time()

        lr = self.trainer.get_lr()
        count = self.batch_count * self.batch_size
        if self.mode is Mode.TEST:
            template = 'lr {} => Batch {} Count {}. Time elapsed: {:.2f} | {:.2f} seconds.'
            p(template.format(lr, self.batch_count, count, batch_time, total_time), debug=False)
        else:
            percentage = (self.interval_correct / max(self.interval_count, 1)) * 100

            template = None
            if self.print_accuracy:
                template = 'lr {0} => Epoch {1} | Batch {2} | Count {3} | Time: {4:.2f}/{5:.2f} | Accuracy: {6:.2f} % | Loss: {7:.6f}/{8:.6f}'
            else:
                template = 'lr {0} => Epoch {1} | Batch {2} | Count {3} | Time: {4:.2f}/{5:.2f} | Loss: {7:.6f}/{8:.6f}'

            i = max([int(i) for i in re.findall('(?<={)\d*(?=[}:])', template)])
            extras = []
            for k, v in self.extra_logs.items():
                i += 1
                v = v[-(len(v) % self.print_interval):]
                template += ' | {}: {}{}{}'.format(k, '{', i, ':.6f}')
                extras.append(np.mean([np.mean(i) for i in v]))

            epoch = self.trainer.epoch_ran + 1 if self.mode is Mode.TRAIN else self.trainer.epoch_ran
            min_loss, mean_loss, _ = self.get_loss()
            p(template.format(lr, epoch, self.batch_count, count, \
                batch_time, total_time, percentage, min_loss, mean_loss, *extras), debug=False)

    def _last_batch(self, lst):
        length = len(lst)
        if length % self.print_interval == 0:
            return lst[-self.print_interval:]
        else:
            return lst[length // self.print_interval * self.print_interval:]

    def get_percentage(self):
        percentage = self.all_correct / max(self.all_count, 1) * 100
        return percentage

    def get_loss(self):
        losses = self._last_batch(self.losses)
        min_loss, mean_loss, total_loss = float('inf'), float('inf'), float('inf')
        if len(losses):
            min_loss = np.asscalar(np.amin(losses))
            mean_loss = np.asscalar(np.mean(losses))
            total_loss = np.asscalar(np.mean(self.losses))
        return min_loss, mean_loss, total_loss
