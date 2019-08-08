import socket
import traceback
import subprocess
from os.path import join
from typing import List

from sqlalchemy.exc import ProgrammingError

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.models import Computer
from mlcomp.db.providers import ComputerProvider, ProjectProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.misc import now
from mlcomp.utils.settings import MODEL_FOLDER, DATA_FOLDER
from mlcomp.utils.io import yaml_load


def sync_directed(source: Computer, target: Computer, folders_excluded: List):
    current_computer = socket.gethostname()
    end = ' --perms  --chmod=777'
    logger = create_logger()
    for folder, excluded in folders_excluded:
        if len(excluded) > 0:
            excluded = excluded[:]
            for i in range(len(excluded)):
                excluded[i] = f'--exclude {excluded[i]}'
            end += ' ' + ' '.join(excluded)

        if current_computer == source.name:
            command = f'rsync -vhru -e ' \
                f'"ssh -p {target.port} -o StrictHostKeyChecking=no" ' \
                f' {folder}/ {target.user}@{target.ip}:{folder}/ ' \
                f'{end}'
        elif current_computer == target.name:
            command = f'rsync -vhru -e ' \
                f'"ssh -p {source.port} -o StrictHostKeyChecking=no" ' \
                f' {source.user}@{source.ip}:{folder}/ {folder}/' \
                f'{end}'
        else:
            command = f'rsync -vhru -e ' \
                f'"ssh -p {target.port} -o StrictHostKeyChecking=no" ' \
                f' {folder}/ {target.user}@{target.ip}:{folder}/ ' \
                f'{end}'

            command = f'ssh -p {source.port} ' \
                f'{source.user}@{source.ip} "{command}"'

        logger.info(command)
        subprocess.check_output(command, shell=True)


class FileSync:
    def sync(self):
        logger = create_logger()
        hostname = socket.gethostname()
        try:
            session = Session.create_session(key='FileSync')

            provider = ComputerProvider(session)
            project_provider = ProjectProvider(session)

            computer = provider.by_name(hostname)
            last_synced = computer.last_synced
            sync_start = now()

            computers = provider.all_with_last_activtiy()
            computers = [
                c for c in computers
                if (now() - c.last_activity).total_seconds() < 10
            ]

            excluded = []
            projects = project_provider.all_last_activity()
            folders_excluded = []
            for p in projects:
                if last_synced is not None and \
                        (p.last_activity is None or
                         p.last_activity < last_synced):
                    continue

                ignore = yaml_load(p.ignore_folders)
                for f in ignore:
                    excluded.append(str(f))

                folders_excluded.append([join(DATA_FOLDER, p.name), excluded])
                folders_excluded.append([join(MODEL_FOLDER, p.name), []])

            for c in computers:
                if c.name != computer.name:
                    computer.syncing_computer = c.name
                    provider.update()

                    sync_directed(c, computer, folders_excluded)

            computer.last_synced = sync_start
            computer.syncing_computer = None
            provider.update()
        except Exception as e:
            if type(e) == ProgrammingError:
                Session.cleanup()
            logger.error(
                traceback.format_exc(), ComponentType.WorkerSupervisor,
                hostname
            )
