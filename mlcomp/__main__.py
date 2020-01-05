from os.path import join
from typing import Tuple

import click
import socket
from multiprocessing import cpu_count

import torch

from mlcomp.migration.manage import migrate as _migrate
from mlcomp import ROOT_FOLDER, IP, PORT, \
    WORKER_INDEX, SYNC_WITH_THIS_COMPUTER, CAN_PROCESS_TASKS
from mlcomp.db.core import Session
from mlcomp.db.enums import DagType, ComponentType, TaskStatus
from mlcomp.db.models import Computer
from mlcomp.db.providers import \
    ComputerProvider, \
    TaskProvider, \
    StepProvider, \
    ProjectProvider
from mlcomp.utils.config import merge_dicts_smart, dict_from_list_str
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.logging import create_logger
from mlcomp.worker.sync import sync_directed
from mlcomp.worker.tasks import execute_by_id
from mlcomp.utils.misc import memory, disk, get_username
from mlcomp.server.back.create_dags import dag_standard, dag_pipe

_session = Session.create_session(key=__name__)


def _dag(config: str, debug: bool = False, control_reqs=True,
         params: Tuple[str] = ()):
    logger = create_logger(_session, name='_dag')
    logger.info('started', ComponentType.Client)

    config_text = open(config, 'r').read()
    config_parsed = yaml_load(config_text)
    params = dict_from_list_str(params)
    config_parsed = merge_dicts_smart(config_parsed, params)
    config_text = yaml_dump(config_parsed)

    logger.info('config parsed', ComponentType.Client)

    type_name = config_parsed['info'].get('type', 'standard')
    if type_name == DagType.Standard.name.lower():
        return dag_standard(
            session=_session,
            config=config_parsed,
            debug=debug,
            config_text=config_text,
            config_path=config,
            control_reqs=control_reqs,
            logger=logger,
            component=ComponentType.Client
        )

    return dag_pipe(
        session=_session, config=config_parsed, config_text=config_text
    )


def _create_computer():
    tot_m, used_m, free_m = memory()
    tot_d, used_d, free_d = disk(ROOT_FOLDER)
    computer = Computer(
        name=socket.gethostname(),
        gpu=torch.cuda.device_count(),
        cpu=cpu_count(),
        memory=tot_m,
        ip=IP,
        port=PORT,
        user=get_username(),
        disk=tot_d,
        root_folder=ROOT_FOLDER,
        sync_with_this_computer=SYNC_WITH_THIS_COMPUTER,
        can_process_tasks=CAN_PROCESS_TASKS
    )
    ComputerProvider(_session).create_or_update(computer, 'name')


@click.group()
def main():
    pass


@main.command()
def migrate():
    _migrate()


@main.command()
@click.argument('config')
@click.option('--control_reqs', type=bool, default=True)
@click.option('--params', multiple=True)
def dag(config: str, control_reqs: bool, params):
    _dag(config, control_reqs=control_reqs, params=params)


@main.command()
@click.argument('config')
@click.option('--debug', type=bool, default=True)
@click.option('--params', multiple=True)
def execute(config: str, debug: bool, params):
    _create_computer()

    # Fail all InProgress Tasks
    logger = create_logger(_session, __name__)

    provider = TaskProvider(_session)
    step_provider = StepProvider(_session)

    for t in provider.by_status(
            TaskStatus.InProgress, worker_index=WORKER_INDEX
    ):
        step = step_provider.last_for_task(t.id)
        logger.error(
            f'Task Id = {t.id} was in InProgress state '
            f'when another tasks arrived to the same worker',
            ComponentType.Worker, t.computer_assigned, t.id, step
        )
        provider.change_status(t, TaskStatus.Failed)

    # Create dag
    created_dag = _dag(config, debug, params=params)
    for ids in created_dag.values():
        for id in ids:
            task = provider.by_id(id)
            task.gpu_assigned = ','.join(
                [str(i) for i in range(torch.cuda.device_count())])

            provider.commit()
            execute_by_id(id, exit=False)


@main.command()
@click.argument('project')
@click.option('--computer', help='sync computer with all the others')
@click.option(
    '--only_from',
    is_flag=True,
    help='only copy files from the computer to all the others'
)
@click.option(
    '--only_to',
    is_flag=True,
    help='only copy files from all the others to the computer'
)
def sync(project: str, computer: str, only_from: bool, only_to: bool):
    _create_computer()

    computer = computer or socket.gethostname()
    provider = ComputerProvider(_session)
    project_provider = ProjectProvider(_session)
    computer = provider.by_name(computer)
    computers = provider.all()
    folders_excluded = []
    p = project_provider.by_name(project)
    assert p, f'Project={project} is not found'

    ignore = yaml_load(p.ignore_folders)
    excluded = []
    for f in ignore:
        excluded.append(str(f))

    folders_excluded.append([join('data', p.name), excluded])
    folders_excluded.append([join('models', p.name), []])

    for c in computers:
        if c.name != computer.name:
            if not only_from:
                sync_directed(_session, computer, c, folders_excluded)
            if not only_to:
                sync_directed(_session, c, computer, folders_excluded)


@main.command()
def init():
    # already done by importing mlcomp
    # that is needed to import it
    pass


if __name__ == '__main__':
    main()
