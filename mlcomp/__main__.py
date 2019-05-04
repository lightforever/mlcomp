import click
from mlcomp.db.providers import *
import os
from mlcomp.task.storage import Storage
from mlcomp.utils.config import load_ordered_yaml
from mlcomp.task.executors import Executor
import json
from mlcomp.task.app import app
import socket
from multiprocessing import cpu_count
import torch

@click.group()
def main():
    pass


@main.command()
def worker():
    provider = ComputerProvider()
    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

    computer = Computer(name=socket.gethostname(), gpu=torch.cuda.device_count(), cpu=cpu_count(), memory=tot_m)
    provider.create_or_update(computer, 'name')

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

@main.command()
@click.argument('config')
def dag(config: str):
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

    created = dict()
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
                    dag = dag.id
                )
                provider.add(task)
                created[k] = task.id

                if 'depends' in v:
                    for d in v['depends']:
                        provider.add_dependency(created[k], created[d])


@main.command()
@click.argument('config')
def execute(config: str):
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
                executor = Executor.from_config(k, config)
                executor()
                created.add(k)


if __name__ == '__main__':
    main()
