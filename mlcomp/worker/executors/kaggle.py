import json
import shutil
from enum import Enum
import os
import time

import socket

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.worker.executors.base.executor import Executor
from mlcomp.utils.logging import create_logger
from mlcomp.utils.config import Config

try:
    from kaggle import api
except OSError:
    logger = create_logger(Session.create_session(), __name__)
    logger.warning(
        'Could not find kaggle.json. '
        'Kaggle executors can not be used',
        ComponentType.Worker, socket.gethostname()
    )


class DownloadType(Enum):
    Kaggle = 0
    Link = 1


@Executor.register
class Download(Executor):
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
    def __init__(
            self,
            competition: str,
            file: str,
            type: str = 'file',
            predict_column: str = None,
            kernel_suffix: str = 'api',
            message: str = '',
            wait_seconds=60 * 20
    ):
        assert type in ['file', 'kernel']

        if type == 'kernel':
            assert predict_column, 'predict_column must be specified'

        self.competition = competition
        self.file = file
        self.message = message
        self.wait_seconds = wait_seconds
        self.type = type
        self.kernel_suffix = kernel_suffix
        self.file_name = os.path.basename(file)
        self.predict_column = predict_column

    def file_submit(self):
        api.competition_submit(
            self.file, message=self.message, competition=self.competition
        )

    def kernel_submit(self):
        folder = 'submit'
        os.makedirs(folder, exist_ok=True)

        shutil.copy(self.file,
                    os.path.join(folder, self.file_name))

        config = api.read_config_file()
        username = config['username']
        dataset_meta = {
            'title': f'{self.competition}-{self.kernel_suffix}-dataset',
            'id': f'{username}/'
                  f'{self.competition}-{self.kernel_suffix}-dataset',
            'licenses': [
                {
                    'name': 'CC0-1.0'
                }
            ]
        }
        with open(f'{folder}/dataset-metadata.json', 'w') as f:
            json.dump(dataset_meta, f)

        try:
            api.dataset_status(dataset_meta['id'])
            api.dataset_create_version(folder, 'Updated')
        except Exception:
            api.dataset_create_new(folder)

        kernel_meta = {
            'id': f'{username}/{self.competition}-'
                  f'{self.kernel_suffix}',
            'title': f'{self.competition}-{self.kernel_suffix}',
            'code_file': 'code.py',
            'language': 'python',
            'kernel_type': 'script',
            'is_private': 'true',
            'enable_gpu': 'false',
            'enable_internet': 'false',
            'dataset_sources': [dataset_meta['id']],
            'competition_sources': [self.competition],
            'name': f'{self.competition}-{self.kernel_suffix}'
        }
        with open(f'{folder}/kernel-metadata.json', 'w') as f:
            json.dump(kernel_meta, f)

        code = '''
import pandas as pd

DATA_DIR = '../input/{self.competition}'
CSV_FILE = '../input/{self.competition}-' + \
           '{self.kernel_suffix}-dataset/{self.file_name}'

df = pd.read_csv(DATA_DIR + '/sample_submission.csv')
df_predict = pd.read_csv(CSV_FILE)

keys = [c for c in df.columns if c!='{self.predict_column}']
predict_values = dict()
for index, row in df_predict.iterrows():
    key = tuple([row[k] for k in keys])
    predict_values[key] = row

res = []
for index, row in df.iterrows():
    key = tuple([row[k] for k in keys])
    if key in predict_values:
        res.append(predict_values[key])
    else:
        res.append(row)

res = pd.DataFrame(res)
res.to_csv('submission.csv', index=False)
        '''.replace('{self.competition}', self.competition).replace(
            '{self.kernel_suffix}', self.kernel_suffix).replace(
            '{self.file_name}', self.file_name).replace(
            '{self.predict_column}', self.predict_column)

        with open(f'{folder}/code.py', 'w') as f:
            f.write(code)

        api.kernels_push(folder)

    def work(self):
        submissions = api.competition_submissions(self.competition)
        submission_refs = {s.ref for s in submissions}

        if self.type == 'file':
            self.file_submit()
        else:
            self.kernel_submit()

        step = 10
        for i in range(int(self.wait_seconds // step)):
            try:
                submissions = api.competition_submissions(self.competition)
                for s in submissions:
                    if s.ref not in submission_refs:
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
            except TypeError:
                pass

            time.sleep(step)
        raise Exception(
            f'Submission is not '
            f'complete after {self.wait_seconds}'
        )

    @classmethod
    def _from_config(
            cls, executor: dict, config: Config, additional_info: dict
    ):
        file = os.path.join(config.data_folder, executor['file'])
        return cls(
            file=file,
            competition=executor['competition'],
            message=executor.get('message', 'no message')
        )


__all__ = ['Download', 'Submit']

if __name__ == '__main__':
    submit = Submit('severstal-steel-defect-detection',
                    file='submissions/resnetunet.csv',
                    type='kernel',
                    predict_column='EncodedPixels'
                    )
    submit.work()
