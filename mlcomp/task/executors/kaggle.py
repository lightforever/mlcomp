from mlcomp.task.executors.base import *
from mlcomp.utils.logging import logger
from enum import Enum
from mlcomp.utils.config import Config
import os

try:
    from kaggle import api
except OSError:
    logger.warning('Could not find kaggle.json. Kaggle executors can not be used')

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
    def _from_config(cls, executor: dict, config: Config, additional_info: dict):
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
    def from_config(cls, executor: dict, config: Config, additional_info: dict):
        file = os.path.join(config.data_folder, executor['file'])
        return cls(file=file, competition=executor['competition'], message=executor.get('message', 'no message'))


if __name__ == '__main__':
    # executor = Download('mlcomp/projects/test/data', competition='digit-recognizer')
    executor = Submit(competition='digit-recognizer', file='mlcomp/projects/test/data/sample_submission.csv',
                      message='test')
    executor.work()
