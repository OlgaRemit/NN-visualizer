import torch.optim as optim
from torch import nn

from model.model import Model
from dataset.dataset import Dataset

from enum import Enum
import numpy as np

from collections import defaultdict

class ProblemType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class Trainer:
    def __init__(self,
                n_epochs,
                train: Dataset,
                val: Dataset,
                model: Model,
                type: ProblemType = ProblemType.REGRESSION,
                lr: float=0.03,
                reg_type: str="None",
                reg_strength:float=0
    ):
        self.n_epochs = n_epochs
        self.train = train
        self.val = val
        self.model = model
        self.type = type
        self.reg_type = reg_type
        self.reg_strength = reg_strength
        self.lr = lr
        self.criterion = nn.MSELoss()  #nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self._logs = dict()
        self._logs['train_loss'] = []
        self._logs['val_loss'] = []


    def train_one_epoch(self):
        self.model.train(True)
        for batch in self.train.gen():
            self.optimizer.zero_grad()
            X, Y_true = batch
            Y_pred  = self.model(X)
            loss = self.criterion(Y_pred, Y_true)
            loss.backward()
            self.optimizer.step()
            self._logs['train_loss'].append(loss.item())
        self._logs['train_batches'] = self.train.total_batches()

    def validate(self):
        self.model.train(False)
        for batch in self.val.gen():
            X, Y_true = batch
            Y_pred  = self.model(X)
            loss = self.criterion(Y_pred, Y_true)
            self._logs['val_loss'].append(loss.item())
        self._logs['val_batches'] = self.val.total_batches()

    def run(self):
        print('Start train')
        for epoch in range(self.n_epochs):
            print('Epoch {}'.format(epoch + 1))
            self.train_one_epoch()
            self.validate()
            print("Mean train loss: {}, val loss: {}".format(
                np.mean(self._logs['train_loss'][-self._logs['train_batches']:]),
                np.mean(self._logs['val_loss'][-self._logs['val_batches']:])
            ))
