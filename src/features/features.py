from abc import ABC, abstractmethod

class Features(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self):
        raise NotImplementedError
