from glob import glob
import os
import logging
from os.path import isdir, join
import hashlib
from typing import List, Tuple
import pkgutil
import sys
import pathspec
import pkg_resources

from sqlalchemy.orm import joinedload

from mlcomp.utils.settings import TASK_FOLDER, DATA_FOLDER
from mlcomp.db.models import *
from mlcomp.db.providers import FileProvider, DagStorageProvider, TaskProvider, \
    DagLibraryProvider

from mlcomp.utils.config import Config
from mlcomp.utils.req import control_requirements, read_lines


logger = logging.getLogger(__name__)


class Storage:
    def __init__(self):
        self.file_provider = FileProvider()
        self.provider = DagStorageProvider()
        self.task_provider = TaskProvider()
        self.library_provider = DagLibraryProvider()

    def copy_from(self, src: int, dag: Dag):
        storages = self.provider.query(DagStorage). \
            filter(DagStorage.dag == src). \
            all()
        libraries = self.library_provider.query(DagLibrary). \
            filter(DagLibrary.dag == src). \
            all()

        s_news = []
        for s in storages:
            s_new = DagStorage(dag=dag.id,
                               file=s.file,
                               path=s.path,
                               is_dir=s.is_dir)
            s_news.append(s_new)
        l_news = []
        for l in libraries:
            l_new = DagLibrary(dag=dag.id,
                               library=l.library,
                               version=l.version)
            l_news.append(l_new)

        self.provider.add_all(s_news)
        self.library_provider.add_all(l_news)

    def upload(self, folder: str, dag: Dag):
        hashs = self.file_provider.hashs(dag.project)
        ignore_file = os.path.join(folder, 'file.ignore.txt')
        if not os.path.exists(ignore_file):
            with open(ignore_file, 'w') as f:
                f.write('')

        ignore_patterns = read_lines(ignore_file)
        ignore_patterns.extend(['log', 'data', '__pycache__'])

        files = []
        spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, ignore_patterns)

        for o in glob(os.path.join(folder, '**'), recursive=True):
            path = os.path.relpath(o, folder)
            if spec.match_file(path) or path == '.':
                continue

            if isdir(o):
                self.provider.add(DagStorage(dag=dag.id,
                                             path=path,
                                             is_dir=True))
                continue
            content = open(o, 'rb').read()
            md5 = hashlib.md5(content).hexdigest()
            if md5 in hashs:
                file_id = hashs[md5]
            else:
                file = File(md5=md5,
                            content=content,
                            project=dag.project,
                            dag=dag.id)
                self.file_provider.add(file)
                file_id = file.id
                hashs[md5] = file.id
                files.append(o)

            self.provider.add(DagStorage(dag=dag.id,
                                         path=path,
                                         file=file_id,
                                         is_dir=False))

        reqs = control_requirements(folder, files=files)
        for name, rel, version in reqs:
            self.library_provider.add(DagLibrary(dag=dag.id,
                                                 library=name,
                                                 version=version))

    def download(self, task: int):
        task = self.task_provider.by_id(task, joinedload(Task.dag_rel))
        folder = join(TASK_FOLDER, str(task.id))
        os.makedirs(folder, exist_ok=True)
        items = self.provider.by_dag(task.dag)
        items = sorted(items, key=lambda x: x[1] is not None)
        for item, file in items:
            path = os.path.join(folder, item.path)
            if item.is_dir:
                os.makedirs(path, exist_ok=True)
            else:
                with open(path, 'wb') as f:
                    f.write(file.content)

        config = Config.from_yaml(task.dag_rel.config)
        info = config['info']
        try:
            data_folder = os.path.join(DATA_FOLDER, info['project'])
            os.makedirs(data_folder, exist_ok=True)

            os.symlink(data_folder, os.path.join(folder, 'data'))
        except FileExistsError:
            pass

        sys.path.insert(0, folder)
        return folder

    def import_folder(self, folder: str, libraries: List[Tuple] = None):
        was_installation = False
        folders = [p for p in glob(f'{folder}/*', recursive=True) if
                   os.path.isdir(p) and not '__pycache__' in p]
        folders += [folder]
        library_names = set(n for n, v in (libraries or []))
        library_versions = {n: v for n, v in (libraries or [])}

        for n in library_names:
            try:
                version = pkg_resources.get_distribution(n).version
                need_install = library_versions[n] != version
            except Exception:
                need_install = True

            if need_install:
                os.system(f'pip install {n}=={library_versions[n]}')
                was_installation = True

        for (module_loader, module_name, ispkg) in pkgutil.iter_modules(
                folders):
            module_loader.find_module(module_name).load_module(module_name)

        return was_installation


__all__ = ['Storage']

if __name__ == '__main__':
    storage = Storage()
    storage.download(77)
