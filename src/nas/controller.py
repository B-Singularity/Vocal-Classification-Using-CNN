from abc import ABC, abstractmethod

class Controller(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def _build_sample(self):
        pass

    @abstractmethod
    def _build_greedy(self):
        pass

    @abstractmethod
    def _build_trainer(self):
        pass

