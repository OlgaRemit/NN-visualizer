import pandas as pd
import numpy as np

class CSVLoader(Loader):
    def __init__(self,
        file: str,
        features: list,
        label: str,
        shuffle: bool=False,
        skip: int=0,
        size: int=None,
        seed: int=None
    ):
        self.file = file
        self.features = features
        self.label = label

        df = pd.read_csv(self.file)

        self._data = df[self.features].values
        self._labels = df[label].values

        self._size = size if size is not None else len(self._data) - skip
        assert skip + self._size <= len(self._data)

        full_index = np.arange(len(self._data))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(full_index)
        self._index = full_index[skip:skip + self._size]

    def get(self, idx: int):
        _idx = self._index[idx]
        return (self._data[_idx], self._labels[_idx])

    def __len__(self):
        return self._size
