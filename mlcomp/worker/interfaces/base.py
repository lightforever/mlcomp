from abc import ABC, abstractmethod


class BaseInterface(ABC):
    _child = dict()

    @abstractmethod
    def __call__(self, x: dict) -> dict:
        pass

    @staticmethod
    def register(cls):
        BaseInterface._child[cls.__name__] = cls
        return cls
