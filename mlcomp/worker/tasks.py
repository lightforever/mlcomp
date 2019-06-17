import socket
from mlcomp.db.enums import ComponentType
from sqlalchemy.orm import joinedload

from mlcomp.db.providers import TaskProvider, DagLibraryProvider, StepProvider
from mlcomp.utils.logging import create_logger
from mlcomp.worker.app import app
from mlcomp.db.models import *
from mlcomp.worker.storage import Storage
import traceback
import sys
from celery.signals import celeryd_after_setup
import pickle
from mlcomp.worker.executors import *
from mlcomp.utils.config import Config

def execute_by_id(id: int, repeat_count=1):
    logger = create_logger()
    provider = TaskProvider()
    library_provider = DagLibraryProvider()
    step_provider = StepProvider()
    storage = Storage()
    task = provider.by_id(id, joinedload(Task.dag_rel))
    dag = task.dag_rel
    executor = None
    try:
        assert dag is not None, 'You must fetch task with dag_rel'

        if task.status >= TaskStatus.InProgress.value and repeat_count >= 1:
            logger.warning(f'Task = {task.id}. Status = {task.status}, before the execute_by_id invocation',
                           ComponentType.Worker)
            return

        hostname = socket.gethostname()
        docker_img = os.getenv('DOCKER_IMG', 'default')
        worker_index = int(os.getenv("WORKER_INDEX", -1))

        # Fail all InProgress Tasks assigned to this worker except that task
        for t in provider.by_status(TaskStatus.InProgress, docker_img=docker_img, worker_index=worker_index):
            if t.id != id:
                step = step_provider.last_for_task(t.id)
                logger.error(f'Task Id = {t.id} was in InProgress state when another tasks arrived to the same worker', ComponentType.Worker, step)
                provider.change_status(t, TaskStatus.Failed)

        task.computer_assigned = hostname
        task.pid = os.getpid()
        task.worker_index = worker_index
        provider.change_status(task, TaskStatus.InProgress)

        if not task.debug:
            folder = storage.download(task=id)
        else:
            folder = os.getcwd()

        libraries = library_provider.dag(task.dag)
        was_installation = storage.import_folder(folder, libraries)
        if was_installation and not task.debug:
            if repeat_count > 0:
                try:
                    queue = f'{hostname}_{docker_img}_{os.getenv("WORKER_INDEX")}'
                    logger.warning(traceback.format_exc(), ComponentType.Worker)
                    execute.apply_async((id, repeat_count - 1), queue=queue)
                except Exception:
                    pass
                finally:
                    sys.exit()
        os.chdir(folder)

        config = Config.from_yaml(dag.config)
        executor_type = config['executors'][task.executor]['type']

        assert Executor.is_registered(executor_type), f'Executor {executor_type} was not found'

        additional_info = pickle.loads(task.additional_info) if task.additional_info else dict()
        executor = Executor.from_config(task.executor, config, additional_info=additional_info)

        executor(task, dag)

        provider.change_status(task, TaskStatus.Success)
    except Exception:
        step = executor.step.id if (executor and executor.step) else None
        logger.error(traceback.format_exc(), ComponentType.Worker, step)
        provider.change_status(task, TaskStatus.Failed)
    finally:
        sys.exit()


@celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ["WORKER_INDEX"] = sender.split('@')[1]


@app.task
def execute(id: int, repeat_count: int = 1):
    execute_by_id(id, repeat_count)


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
    logger = create_logger()
    provider = TaskProvider()
    if task.status > TaskStatus.InProgress.value:
        return task.status

    try:
        app.control.revoke(task.celery_id, terminate=True)
    except Exception:
        logger.error(traceback.format_exc(), ComponentType.API)
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
