import time
import socket
import json
import os
import traceback
from multiprocessing import cpu_count

import click
import GPUtil
import psutil
import numpy as np
import torch
from mlcomp.utils.io import yaml_load

from mlcomp import ROOT_FOLDER, MASTER_PORT_RANGE, CONFIG_FOLDER, \
    DOCKER_IMG, DOCKER_MAIN, IP, PORT, WORKER_USAGE_INTERVAL, \
    SYNC_WITH_THIS_COMPUTER, CAN_PROCESS_TASKS
from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus
from mlcomp.utils.logging import create_logger
from mlcomp.db.providers import DockerProvider, TaskProvider
from mlcomp.utils.schedule import start_schedule
from mlcomp.utils.misc import dict_func, now, disk, get_username, \
    kill_child_processes
from mlcomp.worker.app import app
from mlcomp.db.providers import ComputerProvider
from mlcomp.db.models import ComputerUsage, Computer, Docker
from mlcomp.utils.misc import memory
from mlcomp.worker.sync import FileSync

_session = Session.create_session(key='worker')


@click.group()
def main():
    pass


def error_handler(f):
    name = f.__name__
    wrapper_vars = {'session': Session.create_session(key=name)}
    wrapper_vars['logger'] = create_logger(wrapper_vars['session'], name)

    hostname = socket.gethostname()

    def wrapper():
        try:
            f(wrapper_vars['session'], wrapper_vars['logger'])
        except Exception as e:
            if Session.sqlalchemy_error(e):
                Session.cleanup(name)

                wrapper_vars['session'] = Session.create_session(key=name)
                wrapper_vars['logger'] = create_logger(wrapper_vars['session'],
                                                       name)

            wrapper_vars['logger'].error(
                traceback.format_exc(), ComponentType.WorkerSupervisor,
                hostname
            )

    return wrapper


@error_handler
def stop_processes_not_exist(session: Session, logger):
    provider = TaskProvider(session)
    hostname = socket.gethostname()
    tasks = provider.by_status(
        TaskStatus.InProgress,
        task_docker_assigned=DOCKER_IMG,
        computer_assigned=hostname
    )
    hostname = socket.gethostname()
    for t in tasks:
        if not psutil.pid_exists(t.pid):
            # tasks can retry the execution
            if (now() - t.last_activity).total_seconds() < 30:
                continue

            os.system(f'kill -9 {t.pid}')
            t.status = TaskStatus.Failed.value
            logger.error(
                f'process with pid = {t.pid} does not exist. '
                f'Set task to failed state',
                ComponentType.WorkerSupervisor, hostname, t.id
            )

            provider.commit()

            additional_info = yaml_load(t.additional_info)
            for p in additional_info.get('child_processes', []):
                logger.info(f'killing child process = {p}')
                os.system(f'kill -9 {p}')


@error_handler
def worker_usage(session: Session, logger):
    provider = ComputerProvider(session)
    docker_provider = DockerProvider(session)

    computer = socket.gethostname()
    docker = docker_provider.get(computer, DOCKER_IMG)
    usages = []

    count = int(10/WORKER_USAGE_INTERVAL)
    count = max(1, count)

    for _ in range(count):
        # noinspection PyProtectedMember
        memory = dict(psutil.virtual_memory()._asdict())

        try:
            gpus = GPUtil.getGPUs()
        except ValueError as err:
            logger.info(f"Active GPUs not found: {err}")
            gpus = []

        usage = {
            'cpu': psutil.cpu_percent(),
            'disk': disk(ROOT_FOLDER)[1],
            'memory': memory['percent'],
            'gpu': [
                {
                    'memory': g.memoryUtil * 100,
                    'load': g.load * 100
                } for g in gpus
            ]
        }

        provider.current_usage(computer, usage)
        usages.append(usage)
        docker.last_activity = now()
        docker_provider.update()

        time.sleep(WORKER_USAGE_INTERVAL)

    usage = json.dumps({'mean': dict_func(usages, np.mean)})
    provider.add(ComputerUsage(computer=computer, usage=usage, time=now()))


@main.command()
@click.argument('number', type=int)
def worker(number):
    """
    Start worker

    :param number: worker index
    """
    name = f'{socket.gethostname()}_{DOCKER_IMG}'
    argv = [
        'worker', '--loglevel=INFO', '-P=solo', f'-n={name}_{number}',
        '-O fair', '-c=1', '--prefetch-multiplier=1', '-Q', f'{name},'
                                                            f'{name}_{number}'
    ]
    app.worker_main(argv)


@main.command()
def worker_supervisor():
    """
    Start worker supervisor.
    This program controls workers ran on the same machine.
    Also, it writes metric of resources consumption.
    """
    host = socket.gethostname()

    logger = create_logger(_session, 'worker_supervisor')
    logger.info('worker_supervisor start',
                ComponentType.WorkerSupervisor,
                host)

    _create_computer()
    _create_docker()

    start_schedule([(stop_processes_not_exist, 10)])

    if DOCKER_MAIN:
        syncer = FileSync()
        start_schedule([(worker_usage, 0)])
        start_schedule([(syncer.sync, 0)])

    name = f'{host}_{DOCKER_IMG}_supervisor'
    argv = [
        'worker', '--loglevel=INFO', '-P=solo', f'-n={name}', '-O fair',
        '-c=1', '--prefetch-multiplier=1', '-Q', f'{name}'
    ]

    logger.info('worker_supervisor run celery',
                ComponentType.WorkerSupervisor,
                host)

    app.worker_main(argv)


@main.command()
@click.option('--daemon', type=bool, default=True,
              help='start supervisord in a daemon mode')
@click.option('--debug', type=bool, default=False,
              help='use source files instead the installed library')
@click.option('--workers', type=int, default=cpu_count(),
              help='count of workers')
@click.option('--log_level', type=str, default='INFO',
              help='log level of supervisord')
def start(daemon: bool, debug: bool, workers: int, log_level: str):
    """
       Start worker_supervisor and workers
    """

    # creating supervisord config
    supervisor_command = 'mlcomp-worker worker-supervisor'
    worker_command = 'mlcomp-worker worker'

    if debug:
        supervisor_command = 'python mlcomp/worker/__main__.py ' \
                             'worker-supervisor'
        worker_command = 'python mlcomp/worker/__main__.py worker'

    daemon_text = 'false' if daemon else 'true'
    text = [
        '[supervisord]', f'nodaemon={daemon_text}', '', '[program:supervisor]',
        f'command={supervisor_command}', 'autostart=true', 'autorestart=true',
        ''
    ]
    for p in range(workers):
        text.append(f'[program:worker{p}]')
        text.append(f'command={worker_command} {p}')
        text.append('autostart=true')
        text.append('autorestart=true')
        text.append('')

    conf = os.path.join(CONFIG_FOLDER, 'supervisord.conf')
    with open(conf, 'w') as f:
        f.writelines('\n'.join(text))

    os.system(f'supervisord ' f'-c {conf} -e {log_level}')


@main.command()
def stop():
    """
    Stop supervisord started by start command
    """
    lines = os.popen('ps -ef | grep supervisord').readlines()
    for line in lines:
        if 'mlcomp/configs/supervisord.conf' not in line:
            continue
        pid = int(line.split()[1])
        kill_child_processes(pid)


def _create_docker():
    docker = Docker(
        name=DOCKER_IMG,
        computer=socket.gethostname(),
        ports='-'.join(list(map(str, MASTER_PORT_RANGE))),
        last_activity=now()
    )
    DockerProvider(_session).create_or_update(docker, 'name', 'computer')


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


if __name__ == '__main__':
    main()
