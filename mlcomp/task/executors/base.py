from abc import ABC, abstractmethod
from enum import Enum
from kaggle import api
from utils.config import Config
from utils.logging import logger
import os
import traceback


class Executor(ABC):
    _child = dict()

    def __call__(self):
        self.work()

    @abstractmethod
    def work(self):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, executor: str, config: Config):
        executor = config['executors'][executor]
        child_class = Executor._child[executor['type']]
        return child_class.from_config(executor, config)

    @staticmethod
    def register(cls):
        Executor._child[cls.__name__] = cls

    @staticmethod
    def is_registered(cls: str):
        return cls in Executor._child


class DownloadType(Enum):
    Kaggle = 0
    Link = 1


@Executor.register
class Download(Executor):
    def __init__(self, output: str, type=DownloadType.Kaggle, competition: str = None, link: str = None):
        if type == DownloadType.Kaggle and competition is None:
            raise Exception('Competition is required for Kaggle')
        self.type = type
        self.competition = competition
        self.link = link
        self.output = output

    def work(self):
        api.competition_download_files(self.competition, self.output)

    @classmethod
    def from_config(cls, executor: dict, config: Config):
        output = os.path.join(config.data_folder, config.get('output', '.'))
        return cls(output=output, competition=executor['competition'])


@Executor.register
class Submit(Executor):
    def __init__(self, competition: str, file: str, message: str):
        self.competition = competition
        self.file = file
        self.message = message

    def work(self):
        api.competition_submit(self.file, message=self.message, competition=self.competition)

    @classmethod
    def from_config(cls, executor: dict, config: Config):
        file = os.path.join(config.data_folder, executor['file'])
        return cls(file=file, competition=executor['competition'], message=executor.get('message', 'no message'))


if __name__ == '__main__':
    # executor = Download('mlcomp/projects/test/data', competition='digit-recognizer')
    executor = Submit(competition='digit-recognizer', file='mlcomp/projects/test/data/sample_submission.csv',
                      message='test')
    executor()
