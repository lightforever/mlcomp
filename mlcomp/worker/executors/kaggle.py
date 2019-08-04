from enum import Enum
import os
import time

from mlcomp.worker.executors.base.executor import Executor
from mlcomp.utils.logging import logger
from mlcomp.utils.config import Config

try:
    from kaggle import api
except OSError:
    logger.warning(
        'Could not find kaggle.json. '
        'Kaggle executors can not be used'
    )


class DownloadType(Enum):
    Kaggle = 0
    Link = 1


@Executor.register
class Download(Executor):
    __syn__ = 'download'

    def __init__(
        self,
        output: str,
        type=DownloadType.Kaggle,
        competition: str = None,
        link: str = None,
    ):
        if type == DownloadType.Kaggle and competition is None:
            raise Exception('Competition is required for Kaggle')
        self.type = type
        self.competition = competition
        self.link = link
        self.output = output

    def work(self):
        api.competition_download_files(self.competition, self.output)

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        output = os.path.join(config.data_folder, config.get('output', '.'))
        return cls(output=output, competition=executor['competition'])


@Executor.register
class Submit(Executor):
    __syn__ = 'submit'

    def __init__(
        self,
        competition: str,
        file: str,
        message: str = '',
        wait_seconds=60 * 10
    ):
        self.competition = competition
        self.file = file
        self.message = message
        self.wait_seconds = wait_seconds

    def work(self):
        api.competition_submit(
            self.file, message=self.message, competition=self.competition
        )
        step = 10
        for i in range(int(self.wait_seconds // step)):
            submissions = api.competition_submissions(self.competition)
            for s in submissions:
                if s.description == self.message:
                    if s.status == 'complete':
                        if s.publicScore is None:
                            raise Exception(
                                'Submission is complete, '
                                'but publicScore is None'
                            )
                        return float(s.publicScore)
                    elif s.status == 'error':
                        raise Exception(
                            f'Submission error '
                            f'on Kaggle: {s.errorDescription}'
                        )

                    break
            time.sleep(step)
        raise Exception(
            f'Submission is not '
            f'complete after {self.wait_seconds}'
        )

    @classmethod
    def from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        file = os.path.join(config.data_folder, executor['file'])
        return cls(
            file=file,
            competition=executor['competition'],
            message=executor.get('message', 'no message')
        )


__all__ = ['Download', 'Submit']
