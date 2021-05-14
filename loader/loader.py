from abc import ABC, abstractmethod

class Loader(ABC):
    @abstractmethod
    def get(self, idx: int):
        pass

    @abstractmethod
    def __len__(self):
        pass
