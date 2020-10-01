from torch.utils.data.sampler import Sampler
import math
import numpy as np


class BucketsRandomBatchSampler(Sampler):
    def reset(self):
        for bucket in self.buckets:
            np.random.shuffle(bucket)

        self.batches = []
        for i, bucket in enumerate(self.buckets):
            if i >= self.num_of_buckets:
                break

            items_count = self.items_count[i]
            for j in range(self.batch_count[i]):
                batch = self.buckets[i][j * items_count:(j + 1) * items_count]
                self.batches.append(batch)

        np.random.shuffle(self.batches)

    def __init__(self,
                 buckets,
                 bucket_widths,
                 fixed_bucket_size=None,
                 max_size=None):
        self.max_size, self.buckets, self.bucket_widths = max_size, buckets, bucket_widths

        self.num_of_buckets = min(len(buckets), len(bucket_widths))
        if fixed_bucket_size is not None:
            self.items_count = [
                fixed_bucket_size for i in range(self.num_of_buckets)
            ]
        elif max_size is not None:
            self.items_count = [
                math.ceil(max_size / bucket_widths[i])
                for i in range(self.num_of_buckets)
            ]
        else:
            raise Exception(
                'Must declare either max_size or fixed_bucket_size.')
        self.batch_count = [
            math.ceil(len(buckets[i]) / self.items_count[i])
            for i in range(self.num_of_buckets)
        ]
        self.total_count = sum(self.batch_count)

    def __iter__(self):
        self.reset()

        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
