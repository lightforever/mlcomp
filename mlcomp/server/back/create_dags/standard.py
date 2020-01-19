from collections import OrderedDict
import os

from mlcomp.contrib.search.grid import grid_cells
from mlcomp.db.core import Session
from mlcomp.db.enums import TaskType, DagType, ComponentType
from mlcomp.db.models import Report, Task, Dag, ReportTasks, TaskDependence
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
            control_reqs: bool = True,
            logger=None,
            component: ComponentType = None
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
        self.logger = logger
        self.component = component

        self.project = None
        self.layouts = None
        self.dag = None
        self.dag_report_id = None
        self.created = None
        self.project_provider = None

    def log_info(self, message: str):
        if self.logger:
            self.logger.info(message, self.component)

    def create_providers(self):
        self.log_info('create_providers')

        self.provider = TaskProvider(self.session)
        self.report_provider = ReportProvider(self.session)
        self.report_tasks_provider = ReportTasksProvider(self.session)
        self.report_layout_provider = ReportLayoutProvider(self.session)
        self.project_provider = ProjectProvider(self.session)

        self.storage = Storage(self.session, logger=self.logger,
                               component=self.component)
        self.dag_provider = DagProvider(self.session)

    def load_base(self):
        self.log_info('load_base')

        project = self.project_provider.by_name(self.info['project'])
        if project is None:
            project = self.project_provider.add_project(self.info['project'])

        self.project = project.id
        self.layouts = self.report_layout_provider.all()

    def create_report(self):
        self.log_info('create_report')

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
        self.log_info('create_dag')

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
        self.log_info('upload')

        if self.upload_files:
            folder = os.path.dirname(os.path.abspath(self.config_path))
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
            computer=self.info.get('computer') or v.get('computer'),
            gpu=gpu,
            gpu_max=gpu_max,
            cpu=v.get('cpu', 1),
            memory=v.get('memory', 0.1),
            dag=self.dag.id,
            debug=self.debug,
            steps=int(v.get('steps', '1')),
            type=task_type
        )
        task.additional_info = yaml_dump(info)
        report = None
        if self.layout_name and task_type == TaskType.Train.value:
            if self.layout_name not in self.layouts:
                raise Exception(f'Unknown report = {v["report"]}')

            report_config = self.layouts[self.layout_name]
            info['report_config'] = report_config

            task.additional_info = yaml_dump(info)
            report = Report(
                config=yaml_dump(report_config),
                name=task.name,
                project=self.project,
                layout=self.layout_name
            )

        return task, report

    def create_tasks(self):
        self.log_info('create_tasks')

        created = OrderedDict()
        executors = self.config['executors']

        tasks = []
        dependencies = []
        reports = []

        while len(created) < len(executors):
            for k, v in executors.items():
                valid = True
                if 'depends' in v:
                    depends = v['depends']
                    if not isinstance(depends, list):
                        depends = [depends]

                    for d in depends:
                        if d == k:
                            raise Exception(f'Executor {k} depends on itself')

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
                            names.append(cell_name)
                            infos.append({'grid_cell': i})
                    else:
                        names.append(v.get('name', k))
                        infos.append({})

                    k_tasks = []
                    for name, info in zip(names, infos):
                        task, report = self.create_task(k, v, name=name,
                                                        info=info)
                        tasks.append(task)
                        k_tasks.append(task)
                        reports.append(report)

                        if 'depends' in v:
                            depends = v['depends']
                            if not isinstance(depends, list):
                                depends = [depends]

                            for d in depends:
                                for dd in created[d]:
                                    dependencies.append((task, dd))
                    created[k] = k_tasks

        not_empty_reports = [r for r in reports if r is not None]
        if len(not_empty_reports) > 0:
            self.provider.bulk_save_objects(not_empty_reports,
                                            return_defaults=True)
            for report, task in zip(reports, tasks):
                if report is not None:
                    task.report = report.id

        self.provider.bulk_save_objects(tasks,
                                        return_defaults=True)

        if len(not_empty_reports) > 0:
            report_tasks = []
            for report, task in zip(reports, tasks):
                if report is not None:
                    report_tasks.append(
                        ReportTasks(report=report.id, task=task.id))
            self.report_tasks_provider.bulk_save_objects(report_tasks)

        dependencies = [
            TaskDependence(task_id=task.id, depend_id=dd.id) for task, dd in
            dependencies
        ]
        self.provider.bulk_save_objects(dependencies)

        for k, v in created.items():
            created[k] = [vv.id for vv in v]
        self.created = created

    def build(self):
        self.create_providers()

        self.load_base()

        self.create_report()

        self.create_dag()

        self.upload()

        self.create_tasks()

        self.log_info('Done')

        return self.created


def dag_standard(
        session: Session,
        config: dict,
        debug: bool,
        config_text: str = None,
        upload_files: bool = True,
        copy_files_from: int = None,
        config_path: str = None,
        control_reqs: bool = True,
        logger=None,
        component: ComponentType = None
):
    builder = DagStandardBuilder(
        session=session,
        config=config,
        debug=debug,
        config_text=config_text,
        upload_files=upload_files,
        copy_files_from=copy_files_from,
        config_path=config_path,
        control_reqs=control_reqs,
        logger=logger,
        component=component
    )
    return builder.build()


__all__ = ['dag_standard']
