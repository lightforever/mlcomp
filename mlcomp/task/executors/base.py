from abc import ABC, abstractmethod
from kaggle import api

from mlcomp.db.models import Step, Task
from mlcomp.utils.config import Config
from mlcomp.utils.logging import logger, logging
import os
import traceback
from mlcomp.db.providers import LogProvider, StepProvider, Log
from mlcomp.utils.misc import now
from mlcomp.db.enums import *
from queue import Queue

class StepWrap:
    def __init__(self, task: Task):
        self.log_provider = LogProvider()
        self.step_provider = StepProvider()
        self.task = task
        self.children = Queue()

    def __enter__(self):
        self.step = self.start(0, 'main')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end(0, 'main')

    def start(self, level: int, name: str):
        step = Step(level=level,
                    name=name,
                    started=now(),
                    status=StepStatus.InProgress.value,
                    task=self.task.id
                    )
        self.step_provider.add(step)
        self.children.put(step)
        return step

    def end(self, level: int, name: str, metrics: dict = None):
        pass

    def log(self, message: str, level: logging.DEBUG):
        self.log_provider.add(Log(
            step=self.step.id,
            message=message,
            level= level,
            time=now()
        ))



class Executor(ABC):
    _child = dict()

    def log(self, message: str, level: logging.DEBUG):
        self.step.log(message=message, level=level)

    def __call__(self, task: Task):
        self.task = task
        self.step = StepWrap(task)
        with self.step:
            self.work()

    @abstractmethod
    def work(self):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, executor: str, config: Config):
        if executor not in config['executors']:
            raise ModuleNotFoundError(f'Executor {executor} has not been found')
        executor = config['executors'][executor]
        child_class = Executor._child[executor['type']]
        return child_class.from_config(executor, config)

    @staticmethod
    def register(cls):
        Executor._child[cls.__name__] = cls
        return cls

    @staticmethod
    def is_registered(cls: str):
        return cls in Executor._child


@Executor.register
class Dummy(Executor):
    def work(self):
        pass

    @classmethod
    def from_config(cls, executor: dict, config: Config):
        return cls()