import ast

import click
from mlcomp.utils.logging import create_logger

from mlcomp.db.providers import *
import os
from mlcomp.worker.storage import Storage
from mlcomp.utils.config import load_ordered_yaml
import socket
from multiprocessing import cpu_count
import GPUtil
from mlcomp.worker.tasks import execute_by_id
from collections import OrderedDict
from mlcomp.utils.misc import memory


@click.group()
def main():
    pass


@main.command()
@click.argument('name')
@click.option('--class_names')
def project(name, class_names):
    if class_names:
        if os.path.exists(class_names):
            class_names = json.load(open(class_names))
        else:
            class_names = {'default': ast.literal_eval(class_names)}
    else:
        class_names = dict()
    provider = ProjectProvider()
    provider.add(name, class_names)


def _dag(config: str, debug: bool = False):
    config_text = open(config, "r").read()
    config_parsed = load_ordered_yaml(config)
    info = config_parsed['info']
    executors = config_parsed['executors']

    provider = TaskProvider()
    storage = Storage()
    dag_provider = DagProvider()
    report_provider = ReportProvider()
    report_tasks_provider = ReportTasksProvider()
    report_scheme_provider = ReportSchemeProvider()

    folder = os.path.dirname(config)
    project = ProjectProvider().by_name(info['project']).id
    dag = dag_provider.add(Dag(config=config_text, project=project,
                               name=info['name'], docker_img=info.get('docker_img')))
    storage.upload(folder, dag)

    created = OrderedDict()
    schemes = report_scheme_provider.all()
    for k, v in config_parsed.get('reports', dict()).items():
        if k not in schemes:
            report_scheme_provider.add_item(k, v)
        else:
            report_scheme_provider.change(k, v)

        schemes[k] = v

    while len(created) < len(executors):
        for k, v in executors.items():
            valid = True
            if 'depends' in k:
                for d in v['depends']:
                    if d not in executors:
                        raise Exception(f'Executor {k} depend on {d} which does not exist')

                    valid = valid and d in created
            if valid:
                task = Task(
                    name=f'{info["name"]}_{k}',
                    executor=k,
                    computer=info.get('computer'),
                    gpu=v.get('gpu', 0),
                    cpu=v.get('cpu', 1),
                    memory=v.get('memory', 0.1),
                    dag=dag.id,
                    debug=debug,
                    steps=int(v.get('steps', '1'))
                )

                if v.get('report'):
                    if v['report'] not in schemes:
                        raise Exception(f'Unknown report = {v["report"]}')
                    report_config = ReportSchemeInfo.union_schemes(v['report'], schemes)
                    task.additional_info = pickle.dumps({'report_config': report_config})
                    provider.add(task)

                    report = Report(config=json.dumps(report_config), name=task.name, project=project)
                    report_provider.add(report)
                    report_tasks_provider.add(ReportTasks(report=report.id, task=task.id))
                else:
                    provider.add(task)

                created[k] = task.id

                if 'depends' in v:
                    for d in v['depends']:
                        provider.add_dependency(created[k], created[d])
    return created


@main.command()
@click.argument('config')
def dag(config: str):
    _dag(config)


def _create_computer():
    tot_m, used_m, free_m = memory()
    computer = Computer(name=socket.gethostname(), gpu=len(GPUtil.getGPUs()),
                        cpu=cpu_count(), memory=tot_m,
                        ip=os.getenv('IP'),
                        port=int(os.getenv('PORT')),
                        user=os.getenv('USER')
                        )
    ComputerProvider().create_or_update(computer, 'name')


@main.command()
@click.argument('config')
def execute(config: str):
    _create_computer()

    # Fail all InProgress Tasks
    logger = create_logger()
    worker_index = int(os.getenv("WORKER_INDEX", -1))

    provider = TaskProvider()
    step_provider = StepProvider()

    for t in provider.by_status(TaskStatus.InProgress, worker_index=worker_index):
        step = step_provider.last_for_task(t.id)
        logger.error(f'Task Id = {t.id} was in InProgress state when another tasks arrived to the same worker', ComponentType.Worker, step)
        provider.change_status(t, TaskStatus.Failed)

    # Create dag
    created_dag = _dag(config, True)

    config = load_ordered_yaml(config)
    executors = config['executors']

    # Execute
    created = set()
    while len(created) < len(executors):
        for k, v in executors.items():
            valid = True
            if 'depends' in k:
                for d in v['depends']:
                    if d not in executors:
                        raise Exception(f'Executor {k} depend on {d} which does not exist')

                    valid = valid and d in created
            if valid:
                execute_by_id(created_dag[k])
                created.add(k)


if __name__ == '__main__':
    main()
