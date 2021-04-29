from abc import ABC, abstractmethod

class Loader(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def get(self, idx: int):
        pass

    @abstractmethod
    def __len__(self):
        pass
