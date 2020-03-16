import hashlib

import re

from mlcomp.utils.config import merge_dicts_smart
from mlcomp.utils.io import yaml_load, yaml_dump

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus
from mlcomp.db.models import Dag, Task, TaskDependence, DagStorage, File
from mlcomp.db.providers import DagProvider, TaskProvider, DagStorageProvider, \
    FileProvider
from mlcomp.utils.misc import now


class DagCopyBuilder:
    def __init__(
            self,
            session: Session,
            dag: int,
            file_changes: str = '',
            dag_suffix: str = '',
            logger=None,
            component: ComponentType = None
    ):
        self.dag = dag
        self.file_changes = file_changes
        self.session = session
        self.logger = logger
        self.component = component
        self.dag_suffix = dag_suffix

        self.dag_db = None

        self.dag_provider = None
        self.task_provider = None
        self.file_provider = None
        self.dag_storage_provider = None

    def log_info(self, message: str):
        if self.logger:
            self.logger.info(message, self.component)

    def create_providers(self):
        self.log_info('create_providers')

        self.dag_provider = DagProvider(self.session)
        self.task_provider = TaskProvider(self.session)
        self.file_provider = FileProvider(self.session)
        self.dag_storage_provider = DagStorageProvider(self.session)

    def create_dag(self):
        dag = self.dag_provider.by_id(self.dag)
        name = dag.name
        if self.dag_suffix:
            name += ' ' + self.dag_suffix
        dag_new = Dag(name=name, created=now(), config=dag.config,
                      project=dag.project, docker_img=dag.docker_img,
                      img_size=0, file_size=0, type=dag.type)
        self.dag_provider.add(dag_new)
        self.dag_db = dag_new

    def find_replace(self, changes: dict, path: str):
        for k, v in changes.items():
            if not re.match(k, path):
                continue
            return v

    def create_tasks(self):
        tasks = self.task_provider.by_dag(self.dag)
        tasks_new = []
        tasks_old = []

        for t in tasks:
            if t.parent:
                continue

            task = Task(name=t.name, status=TaskStatus.NotRan.value,
                        computer=t.computer, gpu=t.gpu, gpu_max=t.gpu_max,
                        cpu=t.cpu, executor=t.executor, memory=t.memory,
                        steps=t.steps, dag=self.dag_db.id, debug=t.debug,
                        type=t.type,
                        )
            task.additional_info = t.additional_info
            tasks_new.append(task)
            tasks_old.append(t)

        self.task_provider.bulk_save_objects(tasks_new, return_defaults=True)
        old2new = {t_old.id: t_new.id for t_new, t_old in
                   zip(tasks_new, tasks_old)}
        dependencies = self.task_provider.get_dependencies(self.dag)
        dependencies_new = []
        for d in dependencies:
            d_new = TaskDependence(task_id=old2new[d.task_id],
                                   depend_id=old2new[d.depend_id])
            dependencies_new.append(d_new)

        self.task_provider.bulk_save_objects(dependencies_new,
                                             return_defaults=False)

        changes = yaml_load(self.file_changes)
        storages = self.dag_storage_provider.by_dag(self.dag)
        storages_new = []

        for s, f in storages:
            if not isinstance(changes, dict):
                continue

            replace = self.find_replace(changes, s.path)
            if replace is not None and f and s.path.endswith('.yml'):
                content = f.content.decode('utf-8')
                data = yaml_load(content)
                data = merge_dicts_smart(data, replace)
                content = yaml_dump(data).encode('utf-8')
                md5 = hashlib.md5(content).hexdigest()
                f = self.file_provider.by_md5(md5)
                if not f:
                    f = File(
                        content=content,
                        created=now(),
                        project=self.dag_db.project,
                        md5=md5,
                        dag=self.dag_db.id
                    )
                self.file_provider.add(f)

            s_new = DagStorage(dag=self.dag_db.id, file=f.id, path=s.path,
                               is_dir=s.is_dir)
            storages_new.append(s_new)

        self.dag_storage_provider.bulk_save_objects(
            storages_new,
            return_defaults=False
        )

    def build(self):
        self.create_providers()
        self.create_dag()
        self.create_tasks()


def dag_copy(session: Session, dag: int, file_changes: str = '',
             dag_suffix: str = ''):
    builder = DagCopyBuilder(session, dag=dag, file_changes=file_changes,
                             dag_suffix=dag_suffix)
    builder.build()


__all__ = ['dag_copy']
