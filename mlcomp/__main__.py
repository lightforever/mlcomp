import click
import os
import socket

import GPUtil

from mlcomp.db.enums import DagType, ComponentType, TaskStatus
from mlcomp.db.models import Computer
from mlcomp.utils.logging import create_logger
from mlcomp.db.providers import *
from multiprocessing import cpu_count

from mlcomp.utils.settings import ROOT_FOLDER
from mlcomp.worker.sync import sync_directed
from mlcomp.worker.tasks import execute_by_id
from mlcomp.utils.misc import memory, yaml_load, disk
from mlcomp.server.back.create_dags import dag_standard, dag_pipe


def _dag(config: str, debug: bool = False):
    config_text = open(config, "r").read()
    config_parsed = yaml_load(config_text)

    type_name = config_parsed['info'].get('type', "standard")
    if type_name == DagType.Standard.name.lower():
        return dag_standard(config_parsed,
                            debug=debug,
                            config_text=config_text)

    return dag_pipe(config_parsed,
                    config_text=config_text)


def _create_computer():
    tot_m, used_m, free_m = memory()
    tot_d, used_d, free_d = disk(ROOT_FOLDER)
    computer = Computer(name=socket.gethostname(),
                        gpu=len(GPUtil.getGPUs()),
                        cpu=cpu_count(), memory=tot_m,
                        ip=os.getenv('IP'),
                        port=int(os.getenv('PORT')),
                        user=os.getenv('USER'),
                        disk=tot_d
                        )
    ComputerProvider().create_or_update(computer, 'name')


@click.group()
def main():
    pass


@main.command()
@click.argument('config')
def dag(config: str):
    _dag(config)


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

    config = yaml_load(file=config)
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


@main.command()
@click.option('--computer', help='sync computer with all the others')
def sync(computer: str):
    computer = computer or socket.gethostname()
    provider = ComputerProvider()
    computer = provider.by_name(computer)
    computers = provider.all()

    for c in computers:
        if c.name != computer.name:
            sync_directed(computer, c)
            sync_directed(c, computer)


if __name__ == '__main__':
    main()
