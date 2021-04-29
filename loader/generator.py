import numpy as np

from abc import ABC
from loader.loader import Loader


class Generator2D(Loader, ABC):
    def __init__(self, noise, size):
        assert 0 <= noise <= 0.5
        self.noise = noise
        self.size = size


class LinearGenerator2D(Generator2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _random = np.random.random((self.size, 2))
        _noise = np.random.random((self.size, 2))
        _raw_data = 2 * _random - 1  # (0,1) -> (-1,1)
        self._labels = _raw_data[:, 0] + _raw_data[:, 1] > 0
        self._data = self.noise * _noise + (1 - self.noise) * _raw_data

    def get(self, idx: int):
        return (self._data[idx], self._labels[idx])

    def __len__(self):
        return self.size



GENERATORS = {
    "LinearGenerator2D" : LinearGenerator2D
}
