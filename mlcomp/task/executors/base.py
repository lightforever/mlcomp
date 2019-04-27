from abc import ABC, abstractmethod
from enum import Enum
from kaggle import api
from mlcomp.utils.config import Config
import os

class Executor(ABC):
    _child = dict()

    @abstractmethod
    def __call__(self):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, base: Config):
        child_class = Executor._child[config['type']]
        return child_class.from_config(config, base)

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

    def __call__(self):
        api.competition_download_files(self.competition, self.output)

    @classmethod
    def from_config(cls, config: dict, base: Config):
        output = os.path.join(base.data_folder, config.get('output', '.'))
        return cls(output=output, competition=config['competition'])

@Executor.register
class Submit(Executor):
    def __init__(self, competition: str, file: str, message: str):
        self.competition = competition
        self.file = file
        self.message = message

    def __call__(self):
        api.competition_submit(self.file, message=self.message, competition=self.competition)

    @classmethod
    def from_config(cls, config: dict, base: Config):
        file = os.path.join(base.data_folder, config['file'])
        return cls(file=file, competition=config['competition'], message=config.get('message', 'no message'))

if __name__ == '__main__':
    # executor = Download('mlcomp/projects/test/data', competition='digit-recognizer')
    executor = Submit(competition='digit-recognizer', file='mlcomp/projects/test/data/sample_submission.csv',
                      message='test')
    executor()
