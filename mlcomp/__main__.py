from os.path import join
from typing import Tuple

import click
import socket
from multiprocessing import cpu_count

import torch

from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.contrib.search.grid import grid_cells
from mlcomp.migration.manage import migrate as _migrate
from mlcomp import ROOT_FOLDER, IP, PORT, \
    WORKER_INDEX, SYNC_WITH_THIS_COMPUTER, CAN_PROCESS_TASKS, CONFIG_FOLDER
from mlcomp.db.core import Session
from mlcomp.db.enums import DagType, ComponentType, TaskStatus
from mlcomp.db.models import Computer
from mlcomp.db.providers import \
    ComputerProvider, \
    TaskProvider, \
    StepProvider, \
    ProjectProvider
from mlcomp.report import create_report, check_statuses
from mlcomp.utils.config import merge_dicts_smart, dict_from_list_str
from mlcomp.utils.logging import create_logger
from mlcomp.worker.sync import sync_directed, correct_folders
from mlcomp.worker.tasks import execute_by_id
from mlcomp.utils.misc import memory, disk, get_username, \
    get_default_network_interface, now
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
        cells = grid_cells(
            config_parsed['grid']) if 'grid' in config_parsed else [None]
        dags = []
        for cell in cells:
            dag = dag_standard(
                session=_session,
                config=config_parsed,
                debug=debug,
                config_text=config_text,
                config_path=config,
                control_reqs=control_reqs,
                logger=logger,
                component=ComponentType.Client,
                grid_cell=cell
            )
            dags.append(dag)

        return dags

    return [
        dag_pipe(
            session=_session, config=config_parsed, config_text=config_text
        )
    ]


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
    check_statuses()
    _dag(config, control_reqs=control_reqs, params=params)


@main.command()
def report():
    create_report()


@main.command()
@click.argument('config')
@click.option('--debug', type=bool, default=True)
@click.option('--params', multiple=True)
def execute(config: str, debug: bool, params):
    check_statuses()
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

    # Create dags
    dags = _dag(config, debug, params=params)
    for dag in dags:
        for ids in dag.values():
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
@click.option(
    '--online',
    is_flag=True,
    help='sync with only online computers'
)
def sync(project: str, computer: str, only_from: bool, only_to: bool,
         online: bool):
    """
    Syncs specified project on this computer with other computers
    """
    check_statuses()
    _create_computer()

    computer = computer or socket.gethostname()
    provider = ComputerProvider(_session)
    project_provider = ProjectProvider(_session)
    computer = provider.by_name(computer)
    computers = provider.all_with_last_activtiy()
    p = project_provider.by_name(project)
    assert p, f'Project={project} is not found'

    sync_folders = yaml_load(p.sync_folders)
    ignore_folders = yaml_load(p.ignore_folders)

    sync_folders = correct_folders(sync_folders, p.name)
    ignore_folders = correct_folders(ignore_folders, p.name)

    if not isinstance(sync_folders, list):
        sync_folders = []
    if not isinstance(ignore_folders, list):
        ignore_folders = []

    folders = [[s, ignore_folders] for s in sync_folders]

    for c in computers:
        if c.name != computer.name:
            if online and (now() - c.last_activity).total_seconds() > 100:
                continue

            if not only_from:
                sync_directed(_session, computer, c, folders)
            if not only_to:
                sync_directed(_session, c, computer, folders)


@main.command()
def init():
    env_path = join(CONFIG_FOLDER, '.env')
    lines = open(env_path).readlines()
    for i in range(len(lines)):
        if 'NCCL_SOCKET_IFNAME' in lines[i]:
            interface = get_default_network_interface()
            if interface:
                lines[i] = f'NCCL_SOCKET_IFNAME={interface}\n'
    open(env_path, 'w').writelines(lines)


if __name__ == '__main__':
    main()
