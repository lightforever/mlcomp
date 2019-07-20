import socket
import traceback
from datetime import timedelta, datetime
import subprocess

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.models import Computer
from mlcomp.db.providers import ComputerProvider
from mlcomp.utils.logging import logger
from mlcomp.utils.settings import MODEL_FOLDER, DATA_FOLDER


def sync_directed(source: Computer, target: Computer):
    folders = [DATA_FOLDER, MODEL_FOLDER]
    current_computer = socket.gethostname()
    for folder in folders:
        command = f'rsync -vhru -e "ssh -p {target.port}" ' \
            f' {folder}/ {target.user}@{target.ip}:{folder}/ ' \
            f'--perms  --chmod=777'

        if current_computer != source.name:
            command = f"ssh -p {source.port} " \
                f"{source.user}@{source.ip} '{command}'"
        subprocess.check_output(command, shell=True)


def sync():
    try:
        session = Session.create_session(key='FileSync')

        provider = ComputerProvider(session)
        computer = provider.by_name(socket.gethostname())
        min_time = (computer.last_synced - timedelta(seconds=30)) \
            if computer.last_synced else datetime.min
        computers = provider.computers_have_succeeded_tasks(min_time)
        computer.last_synced = datetime.now()

        for c in computers:
            if c.name != computer.name:
                sync_directed(c, computer)

        provider.update()
    except:
        logger.error(traceback.format_exc(),
                     ComponentType.WorkerSupervisor)
