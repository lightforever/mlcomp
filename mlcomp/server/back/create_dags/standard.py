from collections import OrderedDict
import os

from mlcomp.contrib.search.grid import grid_cells
from mlcomp.db.providers import *
from mlcomp.db.enums import TaskType, DagType
from mlcomp.db.models import *
from mlcomp.utils.misc import now
from mlcomp.worker.executors import Executor
from mlcomp.worker.storage import Storage
from mlcomp.utils.io import yaml_dump


class DagStandardBuilder:
    def __init__(self,
                 config: dict,
                 debug: bool,
                 config_text: str = None,
                 upload_files: bool = True,
                 copy_files_from: int = None):
        self.config = config
        self.debug = debug
        self.config_text = config_text
        self.upload_files = upload_files
        self.copy_files_from = copy_files_from

        self.info = config['info']
        self.report_name = self.info.get('report')

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

    def create_providers(self):
        self.provider = TaskProvider()
        self.report_provider = ReportProvider()
        self.report_tasks_provider = ReportTasksProvider()
        self.report_layout_provider = ReportLayoutProvider()

        self.storage = Storage()
        self.dag_provider = DagProvider()

    def load_base(self):
        self.project = ProjectProvider().by_name(self.info['project']).id
        self.layouts = self.report_layout_provider.all()

    def create_report(self):
        self.dag_report_id = None
        report_name = self.report_name
        if report_name:
            if report_name not in self.layouts:
                raise Exception(f'Unknown report = {report_name}')

            report = Report(
                config=yaml_dump(self.layouts[report_name]),
                name=self.info['name'],
                project=self.project,
                layout=report_name
            )
            self.report_provider.add(report)
            self.dag_report_id = report.id

    def create_dag(self):
        dag = Dag(config=self.config_text or yaml_dump(self.config),
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
            self.storage.upload(folder, self.dag)
        elif self.copy_files_from:
            self.storage.copy_from(self.copy_files_from, self.dag)

    def create_task(self,
                    k: str,
                    v: dict,
                    name: str,
                    info: dict):
        task_type = TaskType.User.value
        if v.get('task_type') == 'train' or \
                Executor.is_trainable(v['type']):
            task_type = TaskType.Train.value

        task = Task(
            name=name,
            executor=k,
            computer=self.info.get('computer'),
            gpu=v.get('gpu', 0),
            cpu=v.get('cpu', 1),
            memory=v.get('memory', 0.1),
            dag=self.dag.id,
            debug=self.debug,
            steps=int(v.get('steps', '1')),
            type=task_type
        )
        task.additional_info = ''

        if self.report_name and task_type == TaskType.Train.value:
            if self.report_name not in self.layouts:
                raise Exception(f'Unknown report = {v["report"]}')

            report_config = self.layouts[self.report_name]
            info['report_config'] = report_config
            task.additional_info = yaml_dump(info)
            self.provider.add(task,
                              commit=False)
            report = Report(config=yaml_dump(report_config),
                            name=task.name,
                            project=self.project,
                            layout=self.report_name
                            )
            self.report_provider.add(report)
            task.report = report.id

            self.report_tasks_provider.add(
                ReportTasks(report=report.id, task=task.id))

            self.report_tasks_provider.add(
                ReportTasks(report=self.dag_report_id, task=task.id))

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
                    for d in v['depends']:
                        if d not in executors:
                            raise Exception(
                                f'Executor {k} depend on {d} '
                                f'which does not exist')

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
                            infos.append({
                                'grid_cell': i
                            })
                    else:
                        names.append(k)
                        infos.append({})

                    ids = []
                    for name, info in zip(names, infos):
                        id = self.create_task(k, v, name=name, info=info)
                        ids.append(id)
                        if 'depends' in v:
                            for d in v['depends']:
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


def dag_standard(config: dict,
                 debug: bool,
                 config_text: str = None,
                 upload_files: bool = True,
                 copy_files_from: int = None):
    builder = DagStandardBuilder(config=config,
                                 debug=debug,
                                 config_text=config_text,
                                 upload_files=upload_files,
                                 copy_files_from=copy_files_from
                                 )
    return builder.build()


__all__ = ['dag_standard']
