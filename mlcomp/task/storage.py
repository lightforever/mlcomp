from glob import glob
import os
import logging
from os.path import isdir
import hashlib
from db.models import *
from db.providers import FileProvider, TaskStorageProvider

logger = logging.getLogger(__name__)

class Storage:
    def __init__(self):
        self.file_provider = FileProvider()
        self.ts_provider = TaskStorageProvider()

    def upload(self, folder: str, task: Task):
        hashs = self.file_provider.hashs(task.project)
        for o in glob(os.path.join(folder, '**'), recursive=True):
            path = os.path.relpath(o, folder)
            if path=='.' or path.startswith('data') or path.startswith('./data'):
                continue

            if isdir(o):
                self.ts_provider.add(TaskStorage(task=task.id, path=path, is_dir=True))
                continue
            content =  open(o, 'rb').read()
            md5 = hashlib.md5(content).hexdigest()
            if md5 in hashs:
                file_id = hashs[md5]
            else:
                file = File(md5=md5, content=content, project=task.project)
                self.file_provider.add(file)
                file_id = file.id

            self.ts_provider.add(TaskStorage(task=task.id, path=path, file=file_id, is_dir=False))

    def download(self, task: int):
        folder = f'/tmp/{task}'
        os.makedirs(folder, exist_ok=True)
        items = self.ts_provider.by_task(task)
        items = sorted(items, key=lambda x: x[1] is not None)
        for item, file in items:
            path = os.path.join(folder, item.path)
            if item.is_dir:
                os.makedirs(path, exist_ok=True)
            else:
                with open(path, 'wb') as f:
                    f.write(file.content)


if __name__=='__main__':
    storage = Storage()
    storage.download(31)