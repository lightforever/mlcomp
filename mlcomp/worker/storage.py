from glob import glob
import os
import logging
from os.path import isdir, join, dirname
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
from mlcomp.db.models import DagStorage, Dag, DagLibrary, File, Task
from mlcomp.utils.misc import now, to_snake
from mlcomp.db.providers import FileProvider, \
    DagStorageProvider, \
    TaskProvider, \
    DagLibraryProvider

from mlcomp.utils.config import Config
from mlcomp.utils.req import control_requirements, read_lines

logger = logging.getLogger(__name__)


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
    def __init__(self, session: Session):
        self.file_provider = FileProvider(session)
        self.provider = DagStorageProvider(session)
        self.task_provider = TaskProvider(session)
        self.library_provider = DagLibraryProvider(session)

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
        ignore_patterns.extend(['log', 'data', 'models', '__pycache__'])

        return pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, ignore_patterns
        )

    def upload(self, folder: str, dag: Dag, control_reqs: bool = True):
        hashs = self.file_provider.hashs(dag.project)

        files = []
        all_files = []
        spec = self._build_spec(folder)

        for o in glob(os.path.join(folder, '**'), recursive=True):
            path = os.path.relpath(o, folder)
            if spec.match_file(path) or path == '.':
                continue

            if isdir(o):
                self.provider.add(
                    DagStorage(dag=dag.id, path=path, is_dir=True)
                )
                continue
            content = open(o, 'rb').read()
            md5 = hashlib.md5(content).hexdigest()

            all_files.append(o)

            if md5 in hashs:
                file_id = hashs[md5]
            else:
                file = File(
                    md5=md5,
                    content=content,
                    project=dag.project,
                    dag=dag.id,
                    created=now()
                )
                self.file_provider.add(file)
                file_id = file.id
                hashs[md5] = file.id
                files.append(o)

            self.provider.add(
                DagStorage(dag=dag.id, path=path, file=file_id, is_dir=False)
            )

        if INSTALL_DEPENDENCIES and control_reqs:
            reqs = control_requirements(folder, files=all_files)
            for name, rel, version in reqs:
                self.library_provider.add(
                    DagLibrary(dag=dag.id, library=name, version=version)
                )

    def download(self, task: int):
        task = self.task_provider.by_id(
            task, joinedload(Task.dag_rel, innerjoin=True)
        )
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
            rel_path = os.path.relpath(os.path.splitext(module.path)[0],
                                       base_folder).replace('/', '.')
            classes = pyclbr.readmodule(rel_path, path=[base_folder])
            for k, v in classes.items():
                if is_valid_class(v):
                    importlib.import_module(relative_name(module.path))
                    return True, was_installation

        return False, was_installation


__all__ = ['Storage']
