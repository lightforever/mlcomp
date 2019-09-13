import json
import shutil
from enum import Enum
import os
import time

import socket

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.providers import ModelProvider
from mlcomp.worker.executors.base.equation import Equation
from mlcomp.worker.executors.base.executor import Executor
from mlcomp.utils.logging import create_logger
from mlcomp.utils.config import Config

try:
    from kaggle import api
except OSError:
    logger = create_logger(Session.create_session(), __name__)
    logger.warning(
        'Could not find kaggle.json. '
        'Kaggle executors can not be used', ComponentType.Worker,
        socket.gethostname()
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
class Submit(Equation):
    def __init__(
        self,
        competition: str,
        submit_type: str = 'file',
        predict_column: str = None,
        kernel_suffix: str = 'api',
        message: str = '',
        wait_seconds=60 * 20,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.competition = competition
        self.wait_seconds = wait_seconds
        self.submit_type = submit_type
        self.kernel_suffix = kernel_suffix
        self.predict_column = predict_column
        self.message = message or f'model_id = {self.model_id}'
        self.file = f'data/submissions/{self.name}.csv'
        self.file_name = os.path.basename(self.file)

        assert self.submit_type in ['file', 'kernel']
        if self.submit_type == 'kernel':
            assert self.predict_column, 'predict_column must be specified'

    def file_submit(self):
        self.info(f'file_submit. file = {self.file} start')
        api.competition_submit(
            self.file, message=self.message, competition=self.competition
        )
        self.info(f'file_submit. file = {self.file} end')

    def kernel_submit(self):
        self.info('kernel_submit updating dataset')

        folder = 'submit'
        os.makedirs(folder, exist_ok=True)

        shutil.copy(self.file, os.path.join(folder, self.file_name))

        config = api.read_config_file()
        username = config['username']
        dataset_meta = {
            'title': f'{self.competition}-{self.kernel_suffix}-dataset',
            'id': f'{username}/'
            f'{self.competition}-{self.kernel_suffix}-dataset',
            'licenses': [{
                'name': 'CC0-1.0'
            }]
        }
        with open(f'{folder}/dataset-metadata.json', 'w') as f:
            json.dump(dataset_meta, f)

        try:
            api.dataset_status(dataset_meta['id'])
            api.dataset_create_version(folder, 'Updated')
            self.info('dataset updated')
        except Exception:
            self.info(f'no dataset on Kaggle. creating new')
            api.dataset_create_new(folder)
            self.info('dataset created on Kaggle')

        seconds_to_sleep = 20
        self.info(f'sleeping {seconds_to_sleep} seconds')
        time.sleep(seconds_to_sleep)

        slug = f'{self.competition}-{self.kernel_suffix}'
        kernel_meta = {
            'id': f'{username}/{slug}',
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
            '{self.kernel_suffix}', self.kernel_suffix
        ).replace('{self.file_name}', self.file_name
                  ).replace('{self.predict_column}', self.predict_column)

        with open(f'{folder}/code.py', 'w') as f:
            f.write(code)

        self.info('kernel data created')
        api.kernels_push(folder)
        self.info('kernel is pushed. waiting for the end of the commit')
        self.info(f'kernel address: https://www.kaggle.com/{username}/{slug}')

        seconds = self.wait_seconds
        for i in range(seconds):
            response = api.kernel_status(username, slug)
            if response['status'] == 'complete':
                self.info(f'kernel has completed successfully. '
                          f'Please go to '
                          f'https://www.kaggle.com/{username}/{slug} '
                          f'and push the button "Submit to the competition"')
                return
            if response['status'] == 'error':
                raise Exception(
                    f'Kernel is failed. Msg = {response["failureMessage"]}'
                )
            time.sleep(1)
            self.wait_seconds -= 1

        self.info(f'kernel is not finished after {seconds}')

    def work(self):
        submissions = api.competition_submissions(self.competition)
        submission_refs = {s.ref for s in submissions}

        if self.submit_type == 'file':
            self.file_submit()
        else:
            self.kernel_submit()

        self.info('waiting for the submission on Kaggle')

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
                            score = float(s.publicScore)
                            if self.model_id:
                                provider = ModelProvider(self.session)
                                model = provider.by_id(self.model_id)
                                model.score_public = score
                                provider.commit()

                            return {'res': score}
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


__all__ = ['Download', 'Submit']

if __name__ == '__main__':
    submit = Submit(
        competition='severstal-steel-defect-detection',
        name='resnetunet',
        submit_type='kernel',
        predict_column='EncodedPixels'
    )
    submit.work()
