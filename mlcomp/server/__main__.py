import os
from multiprocessing import cpu_count

import click

from mlcomp import CONFIG_FOLDER, REDIS_PORT, REDIS_PASSWORD
from mlcomp.migration.manage import migrate
from mlcomp.server.back.app import start_server as _start_server
from mlcomp.server.back.app import stop_server as _stop_server
from mlcomp.utils.misc import kill_child_processes


@click.group()
def main():
    pass


@main.command()
def start_site():
    migrate()
    _start_server()


@main.command()
def stop_site():
    _stop_server()


@main.command()
@click.option('--daemon', type=bool, default=True)
@click.option('--debug', type=bool, default=False)
@click.option('--workers', type=int, default=cpu_count())
@click.option('--log_level', type=str, default='INFO')
def start(daemon: bool, debug: bool, workers: int, log_level: str):
    migrate()

    # creating supervisord config
    supervisor_command = 'mlcomp-worker worker-supervisor'
    worker_command = 'mlcomp-worker worker'
    server_command = 'mlcomp-server start-site'

    if debug:
        supervisor_command = 'python mlcomp/worker/__main__.py ' \
                             'worker-supervisor'
        worker_command = 'python mlcomp/worker/__main__.py worker'
        server_command = 'python mlcomp/server/__main__.py start-site'

    folder = os.path.dirname(os.path.dirname(__file__))
    redis_path = os.path.join(folder, 'bin/redis-server')

    daemon_text = 'true' if daemon else 'false'
    text = [
        '[supervisord]', f'nodaemon={daemon_text}', '',
        '[program:supervisor]',
        f'command={supervisor_command}', 'autostart=true', 'autorestart=true',
        '', '[program:redis]', f'command={redis_path} --port {REDIS_PORT}'
                               f' --requirepass {REDIS_PASSWORD}',
        'autostart=true',
        'autorestart=true', '',
        '[program:server]',
        f'command={server_command}',
        'autostart=true',
        'autorestart=true', ''
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
    lines = os.popen('ps -ef | grep supervisord').readlines()
    for line in lines:
        if 'mlcomp/configs/supervisord.conf' not in line:
            continue
        pid = int(line.split()[1])
        kill_child_processes(pid)


if __name__ == '__main__':
    main()
