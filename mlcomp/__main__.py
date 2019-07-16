import ast
import click
import os
import socket
import json

import GPUtil

from mlcomp.db.enums import DagType, ComponentType, TaskStatus
from mlcomp.db.models import Computer
from mlcomp.utils.logging import create_logger
from mlcomp.db.providers import *
from mlcomp.utils.config import load_ordered_yaml
from multiprocessing import cpu_count
from mlcomp.worker.tasks import execute_by_id
from mlcomp.utils.misc import memory
from mlcomp.server.back.create_dags import dag_standard, dag_pipe


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

    if config_parsed['info'].get('type', "Standard") == DagType.Standard.name:
        return dag_standard(config_parsed,
                            debug=debug,
                            config_text=config_text)

    return dag_pipe(config_parsed,
                    config_text=config_text)


@main.command()
@click.argument('config')
def dag(config: str):
    _dag(config)


def _create_computer():
    tot_m, used_m, free_m = memory()
    computer = Computer(name=socket.gethostname(),
                        gpu=len(GPUtil.getGPUs()),
                        cpu=cpu_count(), memory=tot_m,
                        ip=os.getenv('IP'),
                        port=int(os.getenv('PORT')),
                        user=os.getenv('USER')
                        )
    ComputerProvider().create_or_update(computer, 'name')


@main.command()
@click.argument('config')
@click.option('--debug', type=bool, default=True)
def execute(config: str, debug: bool):
    _create_computer()

    # Fail all InProgress Tasks
    logger = create_logger()
    worker_index = int(os.getenv("WORKER_INDEX", -1))

    provider = TaskProvider()
    step_provider = StepProvider()

    for t in provider.by_status(TaskStatus.InProgress,
                                worker_index=worker_index):
        step = step_provider.last_for_task(t.id)
        logger.error(
            f'Task Id = {t.id} was in InProgress state '
            f'when another tasks arrived to the same worker',
            ComponentType.Worker, step)
        provider.change_status(t, TaskStatus.Failed)

    # Create dag
    created_dag = _dag(config, debug)

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
                        raise Exception(
                            f'Executor {k} depend on {d} which does not exist')

                    valid = valid and d in created
            if valid:
                execute_by_id(created_dag[k])
                created.add(k)


if __name__ == '__main__':
    main()
