import os
import socket
import traceback
import subprocess
from os.path import join
from typing import List

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.models import Computer, TaskSynced
from mlcomp.db.providers import ComputerProvider, \
    TaskSyncedProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.misc import now
from mlcomp.utils.io import yaml_load


def sync_directed(
        session: Session, source: Computer, target: Computer,
        folders_excluded: List
):
    current_computer = socket.gethostname()
    end = ' --perms  --chmod=777'
    logger = create_logger(session, __name__)
    for folder, excluded in folders_excluded:
        if len(excluded) > 0:
            excluded = excluded[:]
            for i in range(len(excluded)):
                excluded[i] = f'--exclude {excluded[i]}'
            end += ' ' + ' '.join(excluded)

        source_folder = join(source.root_folder, folder)
        target_folder = join(target.root_folder, folder)

        if current_computer == source.name:
            command = f'rsync -vhru -e ' \
                      f'"ssh -p {target.port} -o StrictHostKeyChecking=no" ' \
                      f'{source_folder}/ ' \
                      f'{target.user}@{target.ip}:{target_folder} {end}'
        elif current_computer == target.name:
            command = f'rsync -vhru -e ' \
                      f'"ssh -p {source.port} -o StrictHostKeyChecking=no" ' \
                      f'{source.user}@{source.ip}:{source_folder}/ ' \
                      f'{target_folder} {end}'
        else:
            command = f'rsync -vhru -e ' \
                      f'"ssh -p {target.port} -o StrictHostKeyChecking=no" ' \
                      f' {source_folder}/ ' \
                      f'{target.user}@{target.ip}:{target_folder}/ {end}'

            command = f'ssh -p {source.port} ' \
                      f'{source.user}@{source.ip} "{command}"'

        logger.info(command, ComponentType.WorkerSupervisor, current_computer)
        subprocess.check_output(command, shell=True)


def copy_remote(
        session: Session, computer_from: str, path_from: str, path_to: str
):
    provider = ComputerProvider(session)
    src = provider.by_name(computer_from)
    host = socket.gethostname()
    if host != computer_from:
        c = f'scp -P {src.port} {src.user}@{src.ip}:{path_from} {path_to}'
    else:
        f'cp {path_from} {path_to}'
    subprocess.check_output(c, shell=True)
    return os.path.exists(path_to)


class FileSync:
    session = Session.create_session(key='FileSync')
    logger = create_logger(session, 'FileSync')

    def sync(self):
        hostname = socket.gethostname()
        try:
            provider = ComputerProvider(self.session)
            task_synced_provider = TaskSyncedProvider(self.session)

            computer = provider.by_name(hostname)
            sync_start = now()

            computers = provider.all_with_last_activtiy()
            computers = [
                c for c in computers
                if (now() - c.last_activity).total_seconds() < 10
            ]
            computers_names = {c.name for c in computers}

            for c, project, tasks in task_synced_provider.for_computer(
                    computer.name):
                if c.name not in computers_names:
                    continue

                if c.syncing_computer:
                    continue

                excluded = list(map(str, yaml_load(project.ignore_folders)))
                folders_excluded = [
                    [join('data', project.name), excluded],
                    [join('models', project.name), []]
                ]

                computer.syncing_computer = c.name
                provider.update()
                sync_directed(self.session, c, computer, folders_excluded)

                for t in tasks:
                    task_synced_provider.add(
                        TaskSynced(computer=computer.name, task=t.id)
                    )

            computer.last_synced = sync_start
            computer.syncing_computer = None
            provider.update()
        except Exception as e:
            if Session.sqlalchemy_error(e):
                Session.cleanup('FileSync')
                self.session = Session.create_session(key='FileSync')
                self.logger = create_logger(self.session, 'FileSync')

            self.logger.error(
                traceback.format_exc(), ComponentType.WorkerSupervisor,
                hostname
            )
