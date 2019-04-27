import click
from mlcomp.db.providers import *
from mlcomp.db.enums import *
import os
from mlcomp.task.storage import Storage
from utils.config import load_ordered_yaml
from mlcomp.task.executors import Executor
import json


@click.group()
def main():
    pass


@main.command()
@click.argument('action', type=click.Choice(['start', 'stop']))
def server(action):
    print(action)


@main.command()
@click.argument('name')
def project(name):
    provider = ProjectProvider()
    provider.add(name)


@main.command()
@click.argument('config')
def task(config: str):
    config = load_ordered_yaml(config)
    info = config['info']
    executors = config['executors']

    provider = TaskProvider()
    storage = Storage()
    folder = os.path.join(os.getcwd(), info['folder'])
    project = ProjectProvider().by_name(info['project']).id

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
                    project=project,
                    name=f'{info["name"]}_{k}',
                    executor=v['type'],
                    config=json.dumps(v)
                )
                provider.add(task)
                storage.upload(folder, task)
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
                executor = Executor.from_config(v, config)
                executor()
                created.add(k)

if __name__ == '__main__':
    main()
