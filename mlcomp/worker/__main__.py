import time
import socket
import json
import os
from multiprocessing import cpu_count

import click
import GPUtil
import psutil
import numpy as np

from mlcomp.utils.settings import ROOT_FOLDER, MASTER_PORT_RANGE
from mlcomp.db.providers import DockerProvider
from mlcomp.utils.schedule import start_schedule
from mlcomp.utils.misc import dict_func, now, disk
from mlcomp.worker.app import app
from mlcomp.db.providers import ComputerProvider
from mlcomp.db.models import ComputerUsage, Computer, Docker
from mlcomp.utils.misc import memory
from mlcomp.worker.sync import FileSync


@click.group()
def main():
    pass


def worker_usage():
    provider = ComputerProvider()
    docker_provider = DockerProvider()

    computer = socket.gethostname()
    docker = docker_provider.get(computer, os.getenv('DOCKER_IMG', 'default'))
    usages = []

    for _ in range(10):
        memory = dict(psutil.virtual_memory()._asdict())

        usage = {
            'cpu': psutil.cpu_percent(),
            'disk': disk(ROOT_FOLDER)[1],
            'memory': memory['percent'],
            'gpu': [{'memory': g.memoryUtil, 'load': g.load}
                    for g in GPUtil.getGPUs()]
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
    argv = [
        'worker',
        '--loglevel=INFO',
        '-P=solo',
        f'-n={number}',
        '-O fair',
        '-c=1',
        '--prefetch-multiplier=1',
        '-Q',
        f'{socket.gethostname()}_{docker_img},'
        f'{socket.gethostname()}_{docker_img}_{number}'
    ]
    print(argv)
    app.worker_main(argv)


@main.command()
def supervisor():
    _create_computer()
    _create_docker()
    if os.getenv('DOCKER_MAIN', 'True') == 'True':
        syncer = FileSync()
        start_schedule([(worker_usage, 0)])
        start_schedule([(syncer.sync, 1)])

    docker_img = os.getenv('DOCKER_IMG', 'default')
    argv = [
        'worker',
        '--loglevel=INFO',
        '-P=solo',
        f'-n=1',
        '-O fair',
        '-c=1',
        '--prefetch-multiplier=1',
        '-Q',
        f'{socket.gethostname()}_{docker_img}_supervisor'
    ]
    app.worker_main(argv)


def _create_docker():
    docker = Docker(
        name=os.getenv('DOCKER_IMG', 'default'),
        computer=socket.gethostname(),
        ports='-'.join(list(map(str, MASTER_PORT_RANGE))),
        last_activity=now()
    )
    DockerProvider().create_or_update(docker, 'name', 'computer')


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


if __name__ == '__main__':
    main()
