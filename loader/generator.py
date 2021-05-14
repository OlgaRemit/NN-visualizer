import numpy as np

from enum import Enum

from loader.loader import Loader

from sklearn.datasets import make_moons, make_circles, make_classification


class GeneratorType(Enum):
    LINEAR = "LINEAR"
    CIRCLES = "CIRCLES"
    MOONS = "MOONS"


class Transform(Enum):
    X1_POWER_2 = "X1_POWER_2"
    X2_POWER_2 = "X2_POWER2"
    X1_X2 = "X1_X2"
    SIN_X1 = "SIN_X1"
    SIN_X2 = "SIN_X2"

def get_fn_from_transform(t: Transform):
    if t == Transform.X1_POWER_2:
        return lambda x: x[:, 0] ** 2
    if t == Transform.X2_POWER_2:
        return lambda x: x[:, 1] ** 2
    if t == Transform.X1_X2:
        return lambda x: x[:, 0] * x[:, 1]
    if t == Transform.SIN_X1:
        return lambda x: np.sin(x[:, 0])
    if t == Transform.SIN_X2:
        return lambda x: np.sin(x[:, 1])


class Generator2D(Loader):
    def __init__(self, noise: float, size: int, type: GeneratorType, shuffle: bool, seed: int):
        assert 0 <= noise <= 0.5
        self.noise = noise
        self.size = size
        self.shuffle = shuffle
        self.seed = seed
        dataset = choose_dataset(type)
        self.data, self.labels = dataset(
            n_samples=self.size,
            noise=self.noise,
            shuffle=self.shuffle,
            random_state=self.seed
        )

    def get(self, idx: int):
        return (self.data[idx], self.labels[idx])

    def __len__(self):
        return self.size


class FeatureTransformGenerator2D(Generator2D):
    def __init__(self, transforms: list, noise: float, size: int, type: GeneratorType, shuffle: bool=False, seed: int=0):
        super().__init__(noise, size, type, shuffle, seed)
        self.aux_data = np.zeros((self.data.shape[0], len(transforms)))
        for i, t in enumerate(transforms):
            f = get_fn_from_transform(t)
            self.aux_data[:, i] = f(self.data)
        self._raw_data = self.data
        self.data = np.concatenate([self._raw_data, self.aux_data], axis=1)


def make_linear(n_samples: int, noise: float, shuffle: bool, random_state: int=0):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=random_state, n_clusters_per_class=1, n_samples=n_samples)
    rng = np.random.RandomState(random_state + 1)
    X += noise * rng.uniform(size=X.shape)
    return X, y

def choose_dataset(type: GeneratorType):
    if type == GeneratorType.LINEAR:
        return make_linear
    if type == GeneratorType.CIRCLES:
        return make_circles
    if type == GeneratorType.MOONS:
        return make_moons
    assert False
