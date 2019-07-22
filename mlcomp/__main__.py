from os.path import join

import click
import os
import socket

import GPUtil

from mlcomp.db.enums import DagType, ComponentType, TaskStatus
from mlcomp.db.models import Computer
from mlcomp.utils.io import yaml_load
from mlcomp.utils.logging import create_logger
from mlcomp.db.providers import *
from multiprocessing import cpu_count

from mlcomp.utils.settings import ROOT_FOLDER, DATA_FOLDER, MODEL_FOLDER
from mlcomp.worker.sync import sync_directed
from mlcomp.worker.tasks import execute_by_id
from mlcomp.utils.misc import memory, disk
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
            ComponentType.Worker,
            t.coputer_assigned,
            t.id,
            step)
        provider.change_status(t, TaskStatus.Failed)

    # Create dag
    created_dag = _dag(config, debug)
    for ids in created_dag.values():
        for id in ids:
            execute_by_id(id)


@main.command()
@click.option('--computer', help='sync computer with all the others')
@click.option('--only_from',
              is_flag=True,
              help='only copy files from the computer to all the others')
@click.option('--only_to',
              is_flag=True,
              help='only copy files from all the others to the computer')
def sync(computer: str, only_from: bool, only_to: bool):
    computer = computer or socket.gethostname()
    provider = ComputerProvider()
    projects = ProjectProvider().all_last_activity()
    computer = provider.by_name(computer)
    computers = provider.all()
    folders_excluded = []
    for p in projects:
        if computer.last_synced is not None and \
                (p.last_activity is None or
                 p.last_activity < computer.last_synced):
            continue

        ignore = yaml_load(p.ignore_folders)
        excluded = []
        for f in ignore:
            excluded.append(str(f))

        folders_excluded.append([join(DATA_FOLDER, p.name), excluded])
        folders_excluded.append([join(MODEL_FOLDER, p.name), []])

    for c in computers:
        if c.name != computer.name:
            if not only_to:
                sync_directed(computer, c, folders_excluded)
            if not only_from:
                sync_directed(c, computer, folders_excluded)


if __name__ == '__main__':
    main()
