from loader.csv_loader import CSVLoader

import pandas as pd


class Session:
    def __init__(self):
        self._build = False
        self.train = None
        self.val = None

    def from_single_csv(self, file: str, features: list, label: str, val_ratio: float, seed: int=None):
        N = count_lines(file)
        N -= 1  # for header
        val_size = int(N * val_ratio)
        train_size = N - val_size

        if seed is None:
            seed = np.random.randint(1e9)

        self.train = CSVLoader(file, features, label, shuffle=True, skip=0, size=train_size, seed=seed)
        self.val = CSVLoader(file, features, label, shuffle=True, skip=train_size, size=val_size, seed=seed)






def count_lines(_file: str):
    with open(_file, 'r') as f:
        return sum(1 for line in f)
