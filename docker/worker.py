import os
import multiprocessing
import click


@click.command()
@click.option('--debug', type=bool, default=False)
@click.option('--cpu_count', type=int, default=None)
def main(debug: bool, cpu_count: int):
    pr = cpu_count or os.getenv('CPU', multiprocessing.cpu_count())
    supervisor_command = 'mlcomp-worker supervisor'
    worker_command = 'mlcomp-worker worker'

    if debug:
        supervisor_command = 'python mlcomp/worker/__main__.py ' \
                             'supervisor'
        worker_command = 'python mlcomp/worker/__main__.py worker'

    text = [
        '[supervisord]',
        'nodaemon=true',
        '',
        '[program:supervisor]',
        f'command={supervisor_command}',
        'autostart=true',
        'autorestart=true',
        ''
    ]
    for p in range(pr):
        text.append(f'[program:worker{p}]')
        text.append(f'command={worker_command} {p}')
        text.append('autostart=true')
        text.append('autorestart=true')
        text.append('')

    with open('supervisord.conf', 'w') as f:
        f.writelines('\n'.join(text))


if __name__ == '__main__':
    main()
