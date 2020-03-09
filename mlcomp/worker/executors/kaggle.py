import json

from mlcomp.utils.io import zip_folder
from typing import List

import shutil
from enum import Enum
import os
import time

import socket

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.utils.misc import du
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
            **kwargs
    ):
        super().__init__(**kwargs)

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
            submit_type: str = 'kernel',
            kernel_suffix: str = 'api',
            message: str = '',
            wait_seconds: int = 60 * 20,
            file: str = None,
            max_size: int = None,
            folders: List[str] = (),
            **kwargs
    ):
        super().__init__(**kwargs)

        if not message and hasattr(self, 'model_id'):
            message = f'model_id = {self.model_id}'

        self.max_size = max_size
        self.competition = competition
        self.wait_seconds = wait_seconds
        self.submit_type = submit_type
        self.kernel_suffix = kernel_suffix
        self.message = message
        self.folders = folders

        if not file and hasattr(self, 'model_name'):
            file = f'data/submissions/{self.model_name}_{self.suffix}.csv'
        self.file = file

        assert self.submit_type in ['file', 'kernel']

    def file_submit(self):
        self.info(f'file_submit. file = {self.file} start')
        api.competition_submit(
            self.file, message=self.message, competition=self.competition
        )
        self.info(f'file_submit. file = {self.file} end')

    def kernel_submit(self):
        self.info('kernel_submit creating dataset')

        folder = os.path.expanduser(
            f'~/.kaggle/competitions/{self.competition}'
        )
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

        total_size = sum([du(f) for f in self.folders])
        if self.max_size:
            assert total_size < self.max_size, \
                f'max_size = {self.max_size} Gb. Current size = {total_size}'

        config = api.read_config_file()
        username = config['username']
        competition = f'deepfake-detection-challenge'
        dataset_meta = {
            'competition': f'{competition}',
            'id': f'{username}/{competition}-api-dataset',
            'licenses': [{
                'name': 'CC0-1.0'
            }],
            'title': 'API auto'
        }
        with open(f'{folder}/dataset-metadata.json', 'w') as f:
            json.dump(dataset_meta, f)

        self.info('\tzipping folders')

        dst = os.path.join(folder, 'dataset.zip')
        zip_folder(folders=self.folders, dst=dst)

        self.info('\tfolders are zipped. uploading dataset')
        if not any(d.ref == dataset_meta['id'] for d in
                   api.dataset_list(user=username)):
            api.dataset_create_new(folder)
        else:
            res = api.dataset_create_version(folder, 'Updated')
            if res.status == 'error':
                raise Exception('dataset_create_version Error: ' + res.error)

        self.info('dataset uploaded. starting kernel')

        # dataset update time
        time.sleep(10)

        slug = 'predict'

        def push_notebook(file: str, slug: str):
            shutil.copy(file, os.path.join(folder, 'predict.ipynb'))

            kernel_meta = {
                'id': f'{username}/{slug}',
                'code_file': 'predict.ipynb',
                'language': 'python',
                'kernel_type': 'notebook',
                'is_private': 'true',
                'enable_gpu': 'true',
                'enable_internet': 'false',
                'dataset_sources': [dataset_meta['id']],
                'competition_sources': [competition],
                'title': f'{slug}',
                'kernel_sources': []
            }
            with open(f'{folder}/kernel-metadata.json', 'w') as f:
                json.dump(kernel_meta, f)

            api.kernels_push(folder)

        push_notebook('predict.ipynb', 'predict')

        self.info('kernel is pushed. waiting for the end of the commit')

        self.info(f'kernel address: https://www.kaggle.com/{username}/{slug}')

        for i in range(10 ** 6):
            response = api.kernel_status(username, slug)
            if response['status'] == 'complete':
                self.info(f'kernel has completed successfully. '
                          f'Pushing predict-full notebook')

                path = '/tmp/predict.ipynb'
                data = json.load(open('predict.ipynb'))
                for cell in data['cells']:
                    if cell['cell_type'] == 'code' \
                            and 'source' in cell \
                            and len(cell['source']) == 1 \
                            and any('max_count' in s for s in cell['source']):
                        cell['source'] = ['max_count = None']

                open(path, 'w').write(json.dumps(data))
                push_notebook(path, 'predict-full')

                self.info(f'Please go to '
                          f'https://www.kaggle.com/{username}/{slug}-full '
                          f'and push the button "Submit to the competition"')
                return

            if response['status'] == 'error':
                raise Exception(
                    f'Kernel is failed. Msg = {response["failureMessage"]}'
                )
            time.sleep(1)

    def work(self):
        if self.submit_type == 'file':
            self.file_submit()
        else:
            self.kernel_submit()


__all__ = ['Download', 'Submit']
