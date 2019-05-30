import click
from mlcomp.db.providers import *
import os
from mlcomp.task.storage import Storage
from mlcomp.utils.config import load_ordered_yaml
from mlcomp.task.app import app
import socket
from multiprocessing import cpu_count
from mlcomp.utils.misc import dict_func
import psutil
import GPUtil
import numpy as np
from mlcomp.task.tasks import execute_by_id
from mlcomp.utils.schedule import start_schedule
from mlcomp.server.back.app import start_server as _start_server
from mlcomp.server.back.app import stop_server as _stop_server


@click.group()
def main():
    pass


def worker_usage():
    provider = ComputerProvider()
    name = socket.gethostname()

    usages = []

    for _ in range(60):
        memory = dict(psutil.virtual_memory()._asdict())

        usage = {
            'cpu': psutil.cpu_percent(),
            'memory': memory['percent'],
            'gpu': [{'memory': g.memoryUtil, 'load': g.load} for g in GPUtil.getGPUs()]
        }

        provider.current_usage(name, usage)
        usage.update(usage)
        usages.append(usage)

    usage = json.dumps({'mean': dict_func(usages, np.mean), 'peak': dict_func(usages, np.max)})
    provider.add(ComputerUsage(computer=name, usage=usage))


@main.command()
@click.argument('number', type=int)
def worker(number):
    docker_img = os.getenv('DOCKER_IMG', 'default')
    argv = [
        'worker',
        '--loglevel=INFO',
        '-P=solo',
        f'-n={number}',
        '-O fair',
        '-c=1',
        '--prefetch-multiplier=1',
        '-Q',
        f'{socket.gethostname()}_{docker_img},{socket.gethostname()}_{docker_img}_{number}'
    ]
    app.worker_main(argv)


@main.command()
def worker_supervisor():
    _create_computer()
    start_schedule([(worker_usage, 60)])

    docker_img = os.getenv('DOCKER_IMG', 'default')
    argv = [
        'worker',
        '--loglevel=INFO',
        '-P=solo',
        f'-n=1',
        '-O fair',
        '-c=1',
        '--prefetch-multiplier=1',
        '-Q',
        f'{socket.gethostname()}_{docker_img}_supervisor'
    ]
    app.worker_main(argv)


@main.command()
def start_server():
    _start_server()


@main.command()
def stop_server():
    _stop_server()


@main.command()
@click.argument('name')
def project(name):
    provider = ProjectProvider()
    provider.add(name)


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

    folder = os.path.join(os.getcwd(), info['folder'])
    project = ProjectProvider().by_name(info['project']).id
    dag = dag_provider.add(Dag(config=config_text, project=project,
                               name=info['name'], docker_img=info.get('docker_img')))
    storage.upload(folder, dag)

    created = OrderedDict()
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
                    debug=debug
                )
                provider.add(task)
                if v.get('report'):
                    if v['report'] not in config_parsed['reports']:
                        raise Exception(f'Unknown report = {v["report"]}')
                    report_config = config_parsed['reports'][v['report']]
                    report = Report(config=json.dumps(report_config), name=task.name, project=project)
                    report_provider.add(report)
                    report_tasks_provider.add(ReportTasks(report=report.id, task=task.id))

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
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    computer = Computer(name=socket.gethostname(), gpu=len(GPUtil.getGPUs()), cpu=cpu_count(), memory=tot_m)
    ComputerProvider().create_or_update(computer, 'name')


@main.command()
@click.argument('config')
def execute(config: str):
    _create_computer()
    created_dag = _dag(config, True)

    config = load_ordered_yaml(config)
    executors = config['executors']

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
