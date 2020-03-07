import os
import socket
import time
import traceback
import subprocess
from os.path import join
from typing import List

from mlcomp import FILE_SYNC_INTERVAL
from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.models import Computer, TaskSynced
from mlcomp.db.providers import ComputerProvider, \
    TaskSyncedProvider, DockerProvider, ProjectProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.misc import now
from mlcomp.utils.io import yaml_load, yaml_dump


def sync_directed(
        session: Session,
        source: Computer,
        target: Computer,
        folders: List
):
    current_computer = socket.gethostname()
    logger = create_logger(session, __name__)
    for folder, excluded in folders:
        end = ' --perms  --chmod=777 --size-only'
        if len(excluded) > 0:
            parts = []
            folder_excluded = False
            for i in range(len(excluded)):
                if excluded[i] == folder:
                    folder_excluded = True
                    break
                if not excluded[i].startswith(folder):
                    continue

                part = os.path.relpath(excluded[i], folder)
                part = f'--exclude {part}'
                parts.append(part)

            if folder_excluded:
                continue

            if len(parts) > 0:
                end += ' ' + ' '.join(parts)

        source_folder = join(source.root_folder, folder)
        target_folder = join(target.root_folder, folder)

        if current_computer == source.name:
            command = f'rsync -vhru -e ' \
                      f'"ssh -p {target.port} -o StrictHostKeyChecking=no" ' \
                      f'{source_folder}/ ' \
                      f'{target.user}@{target.ip}:{target_folder}/ {end}'
        elif current_computer == target.name:
            command = f'rsync -vhru -e ' \
                      f'"ssh -p {source.port} -o StrictHostKeyChecking=no" ' \
                      f'{source.user}@{source.ip}:{source_folder}/ ' \
                      f'{target_folder}/ {end}'
        else:
            command = f'rsync -vhru -e ' \
                      f'"ssh -p {target.port} -o StrictHostKeyChecking=no" ' \
                      f' {source_folder}/ ' \
                      f'{target.user}@{target.ip}:{target_folder}/ {end}'

            command = f'ssh -p {source.port} ' \
                      f'{source.user}@{source.ip} "{command}"'

        logger.info(command, ComponentType.WorkerSupervisor, current_computer)
        try:
            subprocess.check_output(command, shell=True,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            raise Exception(exc.output)


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


def correct_folders(sync_folders: List[str], project_name: str):
    for i in range(len(sync_folders)):
        s = sync_folders[i]
        parts = s.split('/')
        if parts[0] in ['data', 'models']:
            if len(parts) == 1 or parts[1] != project_name:
                parts[0] = join(parts[0], project_name)
        sync_folders[i] = '/'.join(parts)
    return sync_folders


class FileSync:
    session = Session.create_session(key='FileSync')
    logger = create_logger(session, 'FileSync')

    def process_error(self, e: Exception):
        if Session.sqlalchemy_error(e):
            Session.cleanup('FileSync')
            self.session = Session.create_session(key='FileSync')
            self.logger = create_logger(self.session, 'FileSync')

        hostname = socket.gethostname()
        self.logger.error(
            traceback.format_exc(), ComponentType.WorkerSupervisor,
            hostname
        )

    def sync_manual(self, computer: Computer, provider: ComputerProvider):
        """
        button sync was clicked manually
        """
        if not computer.meta:
            return

        meta = yaml_load(computer.meta)
        if 'manual_sync' not in meta:
            return

        manual_sync = meta['manual_sync']

        project_provider = ProjectProvider(self.session)
        docker_provider = DockerProvider(self.session)

        dockers = docker_provider.get_online()
        project = project_provider.by_id(manual_sync['project'])
        sync_folders = manual_sync['sync_folders']
        ignore_folders = manual_sync['ignore_folders']

        sync_folders = correct_folders(sync_folders, project.name)
        ignore_folders = correct_folders(ignore_folders, project.name)

        if not isinstance(sync_folders, list):
            sync_folders = []
        if not isinstance(ignore_folders, list):
            ignore_folders = []

        for docker in dockers:
            if docker.computer == computer.name:
                continue

            source = provider.by_name(docker.computer)
            folders = [[s, ignore_folders] for s in sync_folders]

            computer.syncing_computer = source.name
            provider.update()

            try:
                sync_directed(
                    self.session,
                    target=computer,
                    source=source,
                    folders=folders
                )
            except Exception as e:
                self.process_error(e)
        del meta['manual_sync']
        computer.meta = yaml_dump(meta)
        provider.update()

    def sync(self):
        hostname = socket.gethostname()
        try:
            provider = ComputerProvider(self.session)
            task_synced_provider = TaskSyncedProvider(self.session)

            computer = provider.by_name(hostname)
            sync_start = now()

            if FILE_SYNC_INTERVAL == 0:
                time.sleep(1)
            else:
                self.sync_manual(computer, provider)

                computers = provider.all_with_last_activtiy()
                computers = [
                    c for c in computers
                    if (now() - c.last_activity).total_seconds() < 10
                ]
                computers_names = {c.name for c in computers}

                for c, project, tasks in task_synced_provider.for_computer(
                        computer.name):
                    if c.sync_with_this_computer:
                        if c.name not in computers_names:
                            self.logger.info(f'Computer = {c.name} '
                                             f'is offline. Can not sync',
                                             ComponentType.WorkerSupervisor,
                                             hostname)
                            continue

                        if c.syncing_computer:
                            continue

                        sync_folders = yaml_load(project.sync_folders)
                        ignore_folders = yaml_load(project.ignore_folders)

                        sync_folders = correct_folders(sync_folders,
                                                       project.name)
                        ignore_folders = correct_folders(ignore_folders,
                                                         project.name)

                        if not isinstance(sync_folders, list):
                            sync_folders = []
                        if not isinstance(ignore_folders, list):
                            ignore_folders = []

                        folders = [[s, ignore_folders] for s in sync_folders]

                        computer.syncing_computer = c.name
                        provider.update()

                        sync_directed(self.session, c, computer, folders)

                    for t in tasks:
                        task_synced_provider.add(
                            TaskSynced(computer=computer.name, task=t.id)
                        )

                    time.sleep(FILE_SYNC_INTERVAL)

            computer.last_synced = sync_start
            computer.syncing_computer = None
            provider.update()
        except Exception as e:
            self.process_error(e)
