from tensorboardX import SummaryWriter
from collections import defaultdict

class Writer:
    def __init__(self):
        self.writer = SummaryWriter()
        self._data = defaultdict(int)

    def add_scalar(self, label, val):
        n_iter = self._data[label]
        self._data[label] = n_iter + 1
        self.writer.add_scalar(label, val, n_iter)
