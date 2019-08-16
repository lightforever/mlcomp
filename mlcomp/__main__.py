from os.path import join

import click
import socket
from multiprocessing import cpu_count

import GPUtil

from mlcomp import ROOT_FOLDER, DATA_FOLDER, MODEL_FOLDER, IP, PORT, \
    WORKER_INDEX
from mlcomp.db.core import Session
from mlcomp.db.enums import DagType, ComponentType, TaskStatus
from mlcomp.db.models import Computer
from mlcomp.db.providers import \
    ComputerProvider, \
    TaskProvider, \
    StepProvider, \
    ProjectProvider
from mlcomp.migration.manage import migrate
from mlcomp.utils.io import yaml_load
from mlcomp.utils.logging import create_logger
from mlcomp.worker.sync import sync_directed
from mlcomp.worker.tasks import execute_by_id
from mlcomp.utils.misc import memory, disk, get_username
from mlcomp.server.back.create_dags import dag_standard, dag_pipe

_session = Session.create_session(key=__name__)


def _dag(config: str, debug: bool = False):
    migrate()

    config_text = open(config, 'r').read()
    config_parsed = yaml_load(config_text)

    type_name = config_parsed['info'].get('type', 'standard')
    if type_name == DagType.Standard.name.lower():
        return dag_standard(
            session=_session,
            config=config_parsed,
            debug=debug,
            config_text=config_text,
            config_path=config
        )

    return dag_pipe(
        session=_session, config=config_parsed, config_text=config_text
    )


def _create_computer():
    tot_m, used_m, free_m = memory()
    tot_d, used_d, free_d = disk(ROOT_FOLDER)
    computer = Computer(
        name=socket.gethostname(),
        gpu=len(GPUtil.getGPUs()),
        cpu=cpu_count(),
        memory=tot_m,
        ip=IP,
        port=PORT,
        user=get_username(),
        disk=tot_d
    )
    ComputerProvider(_session).create_or_update(computer, 'name')


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
    logger = create_logger(_session)

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
    created_dag = _dag(config, debug)
    for ids in created_dag.values():
        for id in ids:
            execute_by_id(id)


# @main.command()
# def describe_execution():
#     task_provider = TaskProvider()
#     auxiliary_provider = AuxiliaryProvider()
#     log_provider = LogProvider()
#
#     tasks = task_provider.all()
#     tasks = sorted(tasks, key=lambda x: x.id)


@main.command()
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
def sync(computer: str, only_from: bool, only_to: bool):
    computer = computer or socket.gethostname()
    provider = ComputerProvider(_session)
    projects = ProjectProvider(_session).all_last_activity()
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
            if not only_from:
                sync_directed(_session, computer, c, folders_excluded)
            if not only_to:
                sync_directed(_session, c, computer, folders_excluded)


if __name__ == '__main__':
    main()
