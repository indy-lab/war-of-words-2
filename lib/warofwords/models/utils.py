from math import ceil, floor

import numpy as np


class BatchHelper:
    def __init__(self, data, batch_size, shuffle, drop_last, seed=None):
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        np.random.seed(seed)

    def __iter__(self):
        if self._shuffle:
            data = np.random.permutation(self._data)
        else:
            data = np.array(self._data)
        bs = self._batch_size
        n = int(len(data) / bs)
        for i in range(n):
            yield data[i * bs : (i + 1) * bs]

        if not self._drop_last:
            last = data[(i + 1) * bs :]
            if len(last) != 0:
                yield last

    def __len__(self):
        N = len(self._data) / self._batch_size
        if self._drop_last:
            return int(floor(N))
        else:
            return int(ceil(N))
