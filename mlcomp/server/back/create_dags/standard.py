from collections import OrderedDict
import os

from mlcomp.contrib.search.grid import grid_cells
from mlcomp.db.core import Session
from mlcomp.db.enums import TaskType, DagType
from mlcomp.db.models import Report, Task, Dag, ReportTasks
from mlcomp.db.providers import TaskProvider, \
    ReportProvider, \
    ReportTasksProvider, \
    ReportLayoutProvider, \
    DagProvider, \
    ProjectProvider
from mlcomp.utils.misc import now
from mlcomp.worker.executors import Executor
from mlcomp.worker.storage import Storage
from mlcomp.utils.io import yaml_dump


class DagStandardBuilder:
    def __init__(
            self,
            session: Session,
            config: dict,
            debug: bool,
            config_text: str = None,
            upload_files: bool = True,
            copy_files_from: int = None,
            config_path: str = None,
            control_reqs: bool = True
    ):
        self.session = session
        self.config = config
        self.debug = debug
        self.config_text = config_text
        self.upload_files = upload_files
        self.copy_files_from = copy_files_from
        self.config_path = config_path
        self.control_reqs = control_reqs

        self.info = config['info']
        self.layout_name = self.info.get('layout')

        self.provider = None
        self.report_provider = None
        self.report_tasks_provider = None
        self.report_layout_provider = None
        self.storage = None
        self.dag_provider = None

        self.project = None
        self.layouts = None
        self.dag = None
        self.dag_report_id = None
        self.created = None
        self.project_provider = None

    def create_providers(self):
        self.provider = TaskProvider(self.session)
        self.report_provider = ReportProvider(self.session)
        self.report_tasks_provider = ReportTasksProvider(self.session)
        self.report_layout_provider = ReportLayoutProvider(self.session)
        self.project_provider = ProjectProvider(self.session)

        self.storage = Storage(self.session)
        self.dag_provider = DagProvider(self.session)

    def load_base(self):
        project = self.project_provider.by_name(self.info['project'])
        if project is None:
            project = self.project_provider.add_project(self.info['project'])

        self.project = project.id
        self.layouts = self.report_layout_provider.all()

    def create_report(self):
        self.dag_report_id = None
        layout_name = self.layout_name
        if layout_name:
            if layout_name not in self.layouts:
                raise Exception(f'Unknown layout = {layout_name}')

            report = Report(
                config=yaml_dump(self.layouts[layout_name]),
                name=self.info['name'],
                project=self.project,
                layout=layout_name
            )
            self.report_provider.add(report)
            self.dag_report_id = report.id

    def create_dag(self):
        dag = Dag(
            config=self.config_text or yaml_dump(self.config),
            project=self.project,
            name=self.info['name'],
            docker_img=self.info.get('docker_img'),
            type=DagType.Standard.value,
            created=now(),
            report=self.dag_report_id
        )

        self.dag = self.dag_provider.add(dag)

    def upload(self):
        if self.upload_files:
            folder = os.getcwd()
            if 'expdir' in self.config['info']:
                path = os.path.dirname(os.path.abspath(self.config_path))
                folder = os.path.abspath(
                    os.path.join(path, self.config['info']['expdir'])
                )
            self.storage.upload(folder, self.dag,
                                control_reqs=self.control_reqs)
        elif self.copy_files_from:
            self.storage.copy_from(self.copy_files_from, self.dag)

    def create_task(self, k: str, v: dict, name: str, info: dict):
        task_type = TaskType.User.value
        if v.get('task_type') == 'train' or \
                Executor.is_trainable(v['type']):
            task_type = TaskType.Train.value

        gpu = str(v.get('gpu', '0'))
        if '-' not in gpu:
            gpu = int(gpu)
            gpu_max = gpu
        else:
            gpu, gpu_max = map(int, gpu.split('-'))

        if gpu == 0 and gpu_max > 0:
            raise Exception(f'Executor = {k} Gpu_max can"t be>0 when gpu=0')

        task = Task(
            name=name,
            executor=k,
            computer=self.info.get('computer'),
            gpu=gpu,
            gpu_max=gpu_max,
            cpu=v.get('cpu', 1),
            memory=v.get('memory', 0.1),
            dag=self.dag.id,
            debug=self.debug,
            steps=int(v.get('steps', '1')),
            type=task_type
        )
        task.additional_info = ''

        if self.layout_name and task_type == TaskType.Train.value:
            if self.layout_name not in self.layouts:
                raise Exception(f'Unknown report = {v["report"]}')

            report_config = self.layouts[self.layout_name]
            info['report_config'] = report_config
            task.additional_info = yaml_dump(info)
            self.provider.add(task, commit=False)
            report = Report(
                config=yaml_dump(report_config),
                name=task.name,
                project=self.project,
                layout=self.layout_name
            )
            self.report_provider.add(report)
            task.report = report.id

            self.report_tasks_provider.add(
                ReportTasks(report=report.id, task=task.id)
            )

            self.report_tasks_provider.add(
                ReportTasks(report=self.dag_report_id, task=task.id)
            )

            self.provider.commit()
        else:
            self.provider.add(task)

        return task.id

    def create_tasks(self):
        created = OrderedDict()
        executors = self.config['executors']

        while len(created) < len(executors):
            for k, v in executors.items():
                valid = True
                if 'depends' in v:
                    depends = v['depends']
                    if not isinstance(depends, list):
                        depends = [depends]

                    for d in depends:
                        if d == k:
                            raise Exception(f'Executor {k} depends ot itself')

                        if d not in executors:
                            raise Exception(
                                f'Executor {k} depend on {d} '
                                f'which does not exist'
                            )

                        valid = valid and d in created
                if valid:
                    names = []
                    infos = []
                    if 'grid' in v:
                        grid = v['grid']
                        cells = grid_cells(grid)
                        for i, (cell, cell_name) in enumerate(cells):
                            name = f'{k} {cell_name}'
                            names.append(name)
                            infos.append({'grid_cell': i})
                    else:
                        names.append(k)
                        infos.append({})

                    ids = []
                    for name, info in zip(names, infos):
                        id = self.create_task(k, v, name=name, info=info)
                        ids.append(id)
                        if 'depends' in v:
                            depends = v['depends']
                            if not isinstance(depends, list):
                                depends = [depends]

                            for d in depends:
                                for dd in created[d]:
                                    self.provider.add_dependency(id, dd)
                    created[k] = ids

        self.created = created

    def build(self):
        self.create_providers()

        self.load_base()

        self.create_report()

        self.create_dag()

        self.upload()

        self.create_tasks()

        return self.created


def dag_standard(
        session: Session,
        config: dict,
        debug: bool,
        config_text: str = None,
        upload_files: bool = True,
        copy_files_from: int = None,
        config_path: str = None,
        control_reqs: bool = True
):
    builder = DagStandardBuilder(
        session=session,
        config=config,
        debug=debug,
        config_text=config_text,
        upload_files=upload_files,
        copy_files_from=copy_files_from,
        config_path=config_path,
        control_reqs=control_reqs
    )
    return builder.build()


__all__ = ['dag_standard']
