from loader.loader import Loader
import torch
import numpy as np

class Dataset:
    def __init__(self, loader: Loader, batch_size: int):
        self.loader = loader
        self.batch_size = batch_size
        self._current_batch_idx = 0

    def gen(self):
        self._current_batch_idx = 0
        for batch in np.arange(0, len(self.loader), self.batch_size):
            X, Y = [], []
            for idx in np.arange(self.batch_size):
                _idx = min(batch + idx, len(self.loader) - 1)
                x, y = self.loader.get(_idx)
                X.append(x)
                Y.append(y)
            X = torch.FloatTensor(X)
            Y = torch.FloatTensor(Y)
            Y = Y.reshape(-1, 1)  # (B) -> (B, 1)
            self._current_batch_idx += 1
            yield (X, Y)

    def total_batches(self):
        return self._current_batch_idx
