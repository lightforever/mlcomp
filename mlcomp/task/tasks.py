import socket

from sqlalchemy.orm import joinedload

from mlcomp.db.providers import TaskProvider, DagLibraryProvider
from mlcomp.task.executors import Executor
from mlcomp.utils.config import Config
from mlcomp.utils.logging import logger
from mlcomp.task.app import app
from mlcomp.db.models import *
from mlcomp.task.storage import Storage
import traceback
import os
import sys


def execute_by_id(id: int):
    provider = TaskProvider()
    library_provider = DagLibraryProvider()
    storage = Storage()
    task = provider.by_id(id, joinedload(Task.dag_rel))
    assert task.dag_rel is not None, 'You must fetch task with dag_rel'
    wdir = os.path.dirname(__file__)

    if task.status >= TaskStatus.InProgress.value:
        logger.info(f'Task = {task.id}. Status = {task.status}, before the execute_by_id invocation')
        return

    try:
        task.computer_assigned = socket.gethostname()
        task.pid = os.getpid()
        provider.change_status(task, TaskStatus.InProgress)
        folder = storage.download(task=id)

        config = Config.from_yaml(task.dag_rel.config)

        executor_type = config['executors'][task.executor]['type']
        libraries = library_provider.dag(task.dag)
        storage.import_folder(os.path.join(os.path.dirname(__file__), 'executors'), reload_dep=False)
        storage.import_folder(folder, libraries)
        assert Executor.is_registered(executor_type), f'Executor {executor_type} was not found'

        executor = Executor.from_config(task.executor, config)
        os.chdir(folder)

        executor(task)

        provider.change_status(task, TaskStatus.Success)
    except Exception:
        logger.error(traceback.format_exc())
        provider.change_status(task, TaskStatus.Failed)
        try:
            sys.exit()
        except Exception:
            pass
    finally:
        os.chdir(wdir)


@app.task
def execute(id: int):
    execute_by_id(id)

@app.task
def kill(pid: int):
    os.system(f'kill -9 {pid}')

def queue_list():
    inspect = app.control.inspect()
    if inspect is None:
        return []
    queues = inspect.active_queues()
    if queues is None:
        return []
    return [queue['name'] for node in queues.values() for queue in node]


def stop(task: Task):
    assert task.dag_rel, 'Dag is not in the task'
    provider = TaskProvider()
    if task.status > TaskStatus.InProgress.value:
        return task.status

    try:
        app.control.revoke(task.celery_id, terminate=True)
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        if task.pid:
            queue = f'{task.computer_assigned}_{task.dag_rel.docker_img or "default"}_supervisor'
            kill.apply_async((task.pid,), queue=queue)
        provider.change_status(task, TaskStatus.Stopped)

    return task.status


if __name__ == '__main__':
    execute(81)
    # from task.tasks import execute
    # execute.delay(42)
