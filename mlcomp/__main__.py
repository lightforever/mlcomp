import click
from mlcomp.db.providers import *
import os
from mlcomp.task.storage import Storage
from mlcomp.utils.config import load_ordered_yaml
from mlcomp.task.executors import Executor
from mlcomp.task.app import app
import socket
from multiprocessing import cpu_count
import torch

from utils.misc import dict_func
from utils.schedule import start_schedule
import psutil
import GPUtil
import numpy as np
from mlcomp.task.tasks import execute_by_id


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
def worker():
    provider = ComputerProvider()
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

    computer = Computer(name=socket.gethostname(), gpu=torch.cuda.device_count(), cpu=cpu_count(), memory=tot_m)
    provider.create_or_update(computer, 'name')

    start_schedule([(worker_usage, 60)])

    argv = [
        'worker',
        '--loglevel=INFO',
        '-Q',
        computer.name
    ]
    app.worker_main(argv)


@main.command()
@click.argument('name')
def project(name):
    provider = ProjectProvider()
    provider.add(name)


def _dag(config: str, debug: bool=False):
    config_text = open(config, "r").read()
    config_parsed = load_ordered_yaml(config)
    info = config_parsed['info']
    executors = config_parsed['executors']

    provider = TaskProvider()
    storage = Storage()
    dag_provider = DagProvider()

    folder = os.path.join(os.getcwd(), info['folder'])
    project = ProjectProvider().by_name(info['project']).id
    dag = dag_provider.add(Dag(config=config_text, project=project, name=info['name']))
    storage.upload(folder, dag)

    created = OrderedDict()
    while len(created) < len(executors):
        for k, v in executors.items():
            valid = True
            if 'depends' in k:
                for d in v['depends']:
                    if d not in executors:
                        raise Exception(f'Executor {k} depend on {d} which does not exist')
                    if not Executor.is_registered(executors[d]['type']):
                        raise Exception(f'Executor {d} has not been registered')

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
                created[k] = task.id

                if 'depends' in v:
                    for d in v['depends']:
                        provider.add_dependency(created[k], created[d])
    return created


@main.command()
@click.argument('config')
def dag(config: str):
    _dag(config)


@main.command()
@click.argument('config')
def execute(config: str):
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
                    if not Executor.is_registered(executors[d]['type']):
                        raise Exception(f'Executor {d} has not been registered')

                    valid = valid and d in created
            if valid:
                execute_by_id(created_dag[k])
                created.add(k)


if __name__ == '__main__':
    main()
