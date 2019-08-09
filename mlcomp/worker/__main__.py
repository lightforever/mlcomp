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
from sqlalchemy.exc import ProgrammingError

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus
from mlcomp.utils.logging import create_logger
from mlcomp.utils.settings import ROOT_FOLDER, MASTER_PORT_RANGE, CONFIG_FOLDER
from mlcomp.db.providers import DockerProvider, TaskProvider
from mlcomp.utils.schedule import start_schedule
from mlcomp.utils.misc import dict_func, now, disk
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
    wrapper_vars = {
        'session': Session.create_session(key=name)
    }
    wrapper_vars['logger'] = create_logger(wrapper_vars['session'])

    hostname = socket.gethostname()

    def wrapper():
        try:
            f(wrapper_vars['session'])
        except Exception as e:
            if type(e) == ProgrammingError:
                Session.cleanup(name)

                wrapper_vars['session'] = Session.create_session(key=name)
                wrapper_vars['logger'] = create_logger(wrapper_vars['session'])

            wrapper_vars['logger'].error(
                traceback.format_exc(), ComponentType.WorkerSupervisor,
                hostname
            )

    return wrapper


@error_handler
def stop_processes_not_exist(session: Session):
    provider = TaskProvider(session)
    docker_img = os.getenv('DOCKER_IMG', 'default')
    hostname = socket.gethostname()
    tasks = provider.by_status(
        TaskStatus.InProgress,
        task_docker_assigned=docker_img,
        computer_assigned=hostname
    )
    for t in tasks:
        if not psutil.pid_exists(t.pid):
            os.system(f'kill -9 {t.pid}')
            t.status = TaskStatus.Failed.value
            provider.commit()


@error_handler
def worker_usage(session: Session):
    provider = ComputerProvider(session)
    docker_provider = DockerProvider(session)

    computer = socket.gethostname()
    docker = docker_provider.get(computer, os.getenv('DOCKER_IMG', 'default'))
    usages = []

    for _ in range(10):
        memory = dict(psutil.virtual_memory()._asdict())

        usage = {
            'cpu': psutil.cpu_percent(),
            'disk': disk(ROOT_FOLDER)[1],
            'memory': memory['percent'],
            'gpu': [
                {
                    'memory': g.memoryUtil * 100,
                    'load': g.load * 100
                } for g in GPUtil.getGPUs()
            ]
        }

        provider.current_usage(computer, usage)
        usages.append(usage)
        docker.last_activity = now()
        docker_provider.update()

        time.sleep(1)

    usage = json.dumps({'mean': dict_func(usages, np.mean)})
    provider.add(ComputerUsage(computer=computer, usage=usage, time=now()))


@main.command()
@click.argument('number', type=int)
def worker(number):
    docker_img = os.getenv('DOCKER_IMG', 'default')
    name = f'{socket.gethostname()}_{docker_img}'
    argv = [
        'worker', '--loglevel=INFO', '-P=solo', f'-n={name}_{number}',
        '-O fair', '-c=1', '--prefetch-multiplier=1', '-Q', f'{name},'
        f'{name}_{number}'
    ]
    app.worker_main(argv)


@main.command()
def worker_supervisor():
    _create_computer()
    _create_docker()

    start_schedule([(stop_processes_not_exist, 2)])

    if os.getenv('DOCKER_MAIN', 'True') == 'True':
        syncer = FileSync()
        start_schedule([(worker_usage, 0)])
        start_schedule([(syncer.sync, 1)])

    docker_img = os.getenv('DOCKER_IMG', 'default')
    argv = [
        'worker', '--loglevel=INFO', '-P=solo', f'-n=1', '-O fair', '-c=1',
        '--prefetch-multiplier=1', '-Q',
        f'{socket.gethostname()}_{docker_img}_supervisor'
    ]
    app.worker_main(argv)


@main.command()
@click.option('--debug', type=bool, default=False)
@click.option('--workers', type=int, default=None)
def supervisor(debug: bool, workers: int = cpu_count()):
    # exporting environment variables
    with open(os.path.join(CONFIG_FOLDER, '.env')) as f:
        for l in f.readlines():
            k, v = l.strip().split('=')
            os.environ[k] = v
    # for debugging
    os.environ['PYTHONPATH'] = '.'

    # creating supervisord config
    supervisor_command = 'mlcomp-worker worker-supervisor'
    worker_command = 'mlcomp-worker worker'

    if debug:
        supervisor_command = 'python mlcomp/worker/__main__.py ' \
                             'worker-supervisor'
        worker_command = 'python mlcomp/worker/__main__.py worker'

    text = [
        '[supervisord]', 'nodaemon=true', '', '[program:supervisor]',
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

    os.system('python2 /usr/bin/supervisord ' f'-c {conf} -e DEBUG')


def _create_docker():
    docker = Docker(
        name=os.getenv('DOCKER_IMG', 'default'),
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
        gpu=len(GPUtil.getGPUs()),
        cpu=cpu_count(),
        memory=tot_m,
        ip=os.getenv('IP'),
        port=int(os.getenv('PORT')),
        user=os.getenv('USER'),
        disk=tot_d
    )
    ComputerProvider(_session).create_or_update(computer, 'name')


if __name__ == '__main__':
    main()
