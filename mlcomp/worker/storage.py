from glob import glob
import os
from os.path import isdir, join
import hashlib
from typing import List, Tuple
import pkgutil
import sys
import pathspec
import pkg_resources
import pyclbr
import importlib

from sqlalchemy.orm import joinedload

from mlcomp import TASK_FOLDER, DATA_FOLDER, MODEL_FOLDER, INSTALL_DEPENDENCIES
from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.models import DagStorage, Dag, DagLibrary, File, Task
from mlcomp.utils.misc import now, to_snake
from mlcomp.db.providers import FileProvider, \
    DagStorageProvider, \
    TaskProvider, \
    DagLibraryProvider, DagProvider

from mlcomp.utils.config import Config
from mlcomp.utils.req import control_requirements, read_lines


def get_super_names(cls: pyclbr.Class):
    res = []
    if isinstance(cls.super, list):
        for s in cls.super:
            if isinstance(s, str):
                res.append(s)
            else:
                res.append(s.name)
                res.extend(get_super_names(s))
    elif isinstance(cls.super, pyclbr.Class):
        res.append(cls.super.name)
        res.extend(get_super_names(cls.super))
    elif isinstance(cls, str):
        res.append(cls)
    return res


class Storage:
    def __init__(self, session: Session, logger=None,
                 component: ComponentType = None,
                 max_file_size: int = 10 ** 5, max_count=10 ** 3):
        self.file_provider = FileProvider(session)
        self.provider = DagStorageProvider(session)
        self.task_provider = TaskProvider(session)
        self.library_provider = DagLibraryProvider(session)
        self.dag_provider = DagProvider(session)

        self.logger = logger
        self.component = component
        self.max_file_size = max_file_size
        self.max_count = max_count

    def log_info(self, message: str):
        if self.logger:
            self.logger.info(message, self.component)

    def copy_from(self, src: int, dag: Dag):
        storages = self.provider.query(DagStorage). \
            filter(DagStorage.dag == src). \
            all()
        libraries = self.library_provider.query(DagLibrary). \
            filter(DagLibrary.dag == src). \
            all()

        s_news = []
        for s in storages:
            s_new = DagStorage(
                dag=dag.id, file=s.file, path=s.path, is_dir=s.is_dir
            )
            s_news.append(s_new)
        l_news = []
        for l in libraries:
            l_new = DagLibrary(
                dag=dag.id, library=l.library, version=l.version
            )
            l_news.append(l_new)

        self.provider.add_all(s_news)
        self.library_provider.add_all(l_news)

    def _build_spec(self, folder: str):
        ignore_file = os.path.join(folder, 'file.ignore.txt')
        if not os.path.exists(ignore_file):
            ignore_patterns = []
        else:
            ignore_patterns = read_lines(ignore_file)
        ignore_patterns.extend(
            ['log', '/data', '/models', '__pycache__', '*.ipynb'])

        return pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, ignore_patterns
        )

    def upload(self, folder: str, dag: Dag, control_reqs: bool = True):
        self.log_info('upload started')
        hashs = self.file_provider.hashs(dag.project)
        self.log_info('hashes are retrieved')

        all_files = []
        spec = self._build_spec(folder)

        files = glob(os.path.join(folder, '**'))
        for file in files[:]:
            path = os.path.relpath(file, folder)
            if spec.match_file(path) or path == '.':
                continue
            if os.path.isdir(file):
                child_files = glob(os.path.join(folder, file, '**'),
                                   recursive=True)
                files.extend(child_files)

        if self.max_count and len(files) > self.max_count:
            raise Exception(f'files count = {len(files)} '
                            f'But max count = {self.max_count}')

        self.log_info('list of files formed')

        folders_to_add = []
        files_to_add = []
        files_storage_to_add = []

        total_size_added = 0

        for o in files:
            path = os.path.relpath(o, folder)
            if spec.match_file(path) or path == '.':
                continue

            if isdir(o):
                folder_to_add = DagStorage(dag=dag.id, path=path, is_dir=True)
                folders_to_add.append(folder_to_add)
                continue
            content = open(o, 'rb').read()
            size = sys.getsizeof(content)
            if self.max_file_size and size > self.max_file_size:
                raise Exception(
                    f'file = {o} has size {size}.'
                    f' But max size is set to {self.max_file_size}')
            md5 = hashlib.md5(content).hexdigest()

            all_files.append(o)

            if md5 not in hashs:
                file = File(
                    md5=md5,
                    content=content,
                    project=dag.project,
                    dag=dag.id,
                    created=now()
                )
                hashs[md5] = file
                files_to_add.append(file)
                total_size_added += size

            file_storage = DagStorage(
                dag=dag.id, path=path, file=hashs[md5],
                is_dir=False)
            files_storage_to_add.append(file_storage)

        self.log_info('inserting DagStorage folders')

        if len(folders_to_add) > 0:
            self.provider.bulk_save_objects(folders_to_add)

        self.log_info('inserting Files')

        if len(files_to_add) > 0:
            self.file_provider.bulk_save_objects(files_to_add,
                                                 return_defaults=True)

        self.log_info('inserting DagStorage Files')

        if len(files_storage_to_add) > 0:
            for file_storage in files_storage_to_add:
                if isinstance(file_storage.file, File):
                    # noinspection PyUnresolvedReferences
                    file_storage.file = file_storage.file.id

            self.provider.bulk_save_objects(files_storage_to_add)

        dag.file_size += total_size_added

        self.dag_provider.update()

        if INSTALL_DEPENDENCIES and control_reqs:
            reqs = control_requirements(folder, files=all_files)
            for name, rel, version in reqs:
                self.library_provider.add(
                    DagLibrary(dag=dag.id, library=name, version=version)
                )

    def download_dag(self, dag: int, folder: str):
        os.makedirs(folder, exist_ok=True)

        items = self.provider.by_dag(dag)
        items = sorted(items, key=lambda x: x[1] is not None)
        for item, file in items:
            path = os.path.join(folder, item.path)
            if item.is_dir:
                os.makedirs(path, exist_ok=True)
            else:
                with open(path, 'wb') as f:
                    f.write(file.content)

    def download(self, task: int):
        task = self.task_provider.by_id(
            task, joinedload(Task.dag_rel, innerjoin=True)
        )
        folder = join(TASK_FOLDER, str(task.id))
        self.download_dag(task.dag, folder)

        config = Config.from_yaml(task.dag_rel.config)
        info = config['info']
        try:
            data_folder = os.path.join(DATA_FOLDER, info['project'])
            os.makedirs(data_folder, exist_ok=True)

            os.symlink(
                data_folder,
                os.path.join(folder, 'data'),
                target_is_directory=True
            )
        except FileExistsError:
            pass

        try:
            model_folder = os.path.join(MODEL_FOLDER, info['project'])
            os.makedirs(model_folder, exist_ok=True)

            os.symlink(
                model_folder,
                os.path.join(folder, 'models'),
                target_is_directory=True
            )
        except FileExistsError:
            pass

        sys.path.insert(0, folder)
        return folder

    def import_executor(
            self,
            folder: str,
            base_folder: str,
            executor: str,
            libraries: List[Tuple] = None
    ):

        sys.path.insert(0, base_folder)

        spec = self._build_spec(folder)
        was_installation = False

        folders = [
            p for p in glob(f'{folder}/*', recursive=True)
            if os.path.isdir(p) and not spec.match_file(p)
        ]
        folders += [folder]
        library_names = set(n for n, v in (libraries or []))
        library_versions = {n: v for n, v in (libraries or [])}

        for n in library_names:
            try:
                version = pkg_resources.get_distribution(n).version
                need_install = library_versions[n] != version
            except Exception:
                need_install = True

            if INSTALL_DEPENDENCIES and need_install:
                os.system(f'pip install {n}=={library_versions[n]}')
                was_installation = True

        def is_valid_class(cls: pyclbr.Class):
            return cls.name == executor or \
                   cls.name.lower() == executor or \
                   to_snake(cls.name) == executor

        def relative_name(path: str):
            rel = os.path.relpath(path, base_folder)
            parts = [str(p).split('.')[0] for p in rel.split(os.sep)]
            return '.'.join(parts)

        for (module_loader, module_name,
             ispkg) in pkgutil.iter_modules(folders):
            module = module_loader.find_module(module_name)
            rel_path = os.path.relpath(
                os.path.splitext(module.path)[0], base_folder
            ).replace('/', '.')
            try:
                classes = pyclbr.readmodule(rel_path, path=[base_folder])
            except Exception:
                continue
            for k, v in classes.items():
                if is_valid_class(v):
                    importlib.import_module(relative_name(module.path))
                    return True, was_installation

        return False, was_installation


__all__ = ['Storage']
