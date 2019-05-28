import socket

from sqlalchemy.orm import joinedload

from mlcomp.db.providers import TaskProvider
from mlcomp.task.executors import Executor
import json
from mlcomp.utils.config import Config
from mlcomp.utils.logging import logger
from mlcomp.task.app import app
from mlcomp.db.models import *
from mlcomp.task.storage import Storage
import sys
import traceback
import os


def execute_by_id(id: int):
    provider = TaskProvider()
    storage = Storage()
    thismodule = sys.modules[__name__]
    task = provider.by_id(id, joinedload(Task.dag_rel))
    assert task.dag_rel is not None, 'You must fetch task with dag_rel'
    wdir = os.path.dirname(__file__)

    try:
        task.computer_assigned = socket.gethostname()
        provider.change_status(task, TaskStatus.InProgress)
        folder = storage.download(task=id)

        config = Config.from_yaml(task.dag_rel.config)

        executor_type = config['executors'][task.executor]['type']
        storage.import_folder(thismodule, folder, executor_type)

        executor = Executor.from_config(task.executor, config)
        os.chdir(folder)

        executor(task)

        provider.change_status(task, TaskStatus.Success)
    except Exception:
        logger.error(traceback.format_exc())
        provider.change_status(task, TaskStatus.Failed)
    finally:
        os.chdir(wdir)


@app.task
def execute(id: int):
    execute_by_id(id)


def stop(id: int):
    provider = TaskProvider()
    task = provider.by_id(id)
    if task.status > TaskStatus.InProgress.value:
        return task.status

    try:
        app.control.revoke(task.celery_id, terminate=True)
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        provider.change_status(task, TaskStatus.Stopped)

    return task.status


if __name__ == '__main__':
    execute(81)
    # from task.tasks import execute
    # execute.delay(42)
