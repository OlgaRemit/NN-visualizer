import numpy as np
import pandas as pd

from trainer.trainer import Trainer, ProblemType
from model.model import Model
from dataset.dataset import Dataset
from loader.generator import FeatureTransformGenerator2D, Transform, GeneratorType
from loader.csv_loader import CSVLoader
from session_args import get_args

DEFAULT_GEN_DATASET_SIZE = 100


class Session:
    def __init__(self, trainer: Trainer, batch_size: int):
        self._build = False
        self.train = None
        self.val = None
        self.trainer = trainer
        self.batch_size = batch_size

    def from_single_csv(self, file: str, features: list, label: str, val_ratio: float, seed: int=None):
        N = count_lines(file)
        N -= 1  # for header
        val_size = int(N * val_ratio)
        train_size = N - val_size

        if seed is None:
            seed = np.random.randint(1e9)

        self.train = CSVLoader(file, features, label, shuffle=True, skip=0, size=train_size, seed=seed)
        self.val = CSVLoader(file, features, label, shuffle=True, skip=train_size, size=val_size, seed=seed)
        return self

    def from_generated_dataset(self, type: str, ratio: float, noise: float, transforms: list):
        transforms = [Transform[t] for t in transforms]
        make_dataset = lambda size: FeatureTransformGenerator2D(
            transforms=transforms,
            noise=noise,
            size=size,
            type=GeneratorType[type]
        )
        self.train = make_dataset(int(DEFAULT_GEN_DATASET_SIZE * (1 - ratio)))
        self.val = make_dataset(int(DEFAULT_GEN_DATASET_SIZE * ratio))
        return self

    def build(self):
        self.train = Dataset(self.train, self.batch_size)
        self.val = Dataset(self.val, self.batch_size)
        self.trainer.set_datasets(self.train, self.val)
        self._build = True
        return self

    def run(self):
        assert self._build, "Session not built yet"
        self.trainer.run()
        return None


def count_lines(_file: str):
    with open(_file, 'r') as f:
        return sum(1 for line in f)


def run_session(args: dict):
    layers = list(map(int,args.layers.split(','))) + [1]
    model = Model(layers)
    trainer = Trainer(model, args.n_epochs, ProblemType.REGRESSION, args.lr)
    session = Session(trainer, args.batch_size)
    if args.from_csv:
        session.from_single_csv(args.dataset, args.features.split(','), args.label, args.val_ratio)
    else:
        session.from_generated_dataset(args.dataset, args.val_ratio, args.noise, args.transforms.split(','))

    session.build().run()
    # print(session.trainer.get_logs())


def run():
    args = get_args()
    run_session(args)
