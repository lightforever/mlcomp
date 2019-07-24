import shutil
import socket
import traceback
import sys
from os.path import join

from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import joinedload
from celery.signals import celeryd_after_setup

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus
from mlcomp.db.models import Task
from mlcomp.db.providers import TaskProvider, \
    DagLibraryProvider, \
    DockerProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.io import yaml_load
from mlcomp.worker.app import app
from mlcomp.worker.storage import Storage
from mlcomp.worker.executors import *
from mlcomp.utils.config import Config
from mlcomp.utils.settings import MODEL_FOLDER, TASK_FOLDER


class ExecuteBuilder:
    def __init__(self, id: int, repeat_count: int = 1):
        self.id = id
        self.repeat_count = repeat_count
        self.logger = create_logger()

        self.provider = None
        self.library_provider = None
        self.storage = None
        self.task = None
        self.dag = None
        self.executor = None
        self.hostname = None
        self.docker_img = None
        self.worker_index = None
        self.executor = None

    def create_base(self):
        self.provider = TaskProvider()
        self.library_provider = DagLibraryProvider()
        self.storage = Storage()

        self.task = self.provider.by_id(self.id, joinedload(Task.dag_rel))
        self.dag = self.task.dag_rel
        self.executor = None
        self.hostname = socket.gethostname()

        self.docker_img = os.getenv('DOCKER_IMG', 'default')
        self.worker_index = int(os.getenv("WORKER_INDEX", -1))

    def check_status(self):
        assert self.dag is not None, 'You must fetch task with dag_rel'

        if self.task.status >= TaskStatus.InProgress.value \
                and self.repeat_count >= 1:
            msg = f'Task = {self.task.id}. Status = {self.task.status}, ' \
                f'before the execute_by_id invocation'
            self.logger.error(msg,
                              ComponentType.Worker)
            return

    def change_status(self):
        self.task.computer_assigned = self.hostname
        self.task.pid = os.getpid()
        self.task.worker_index = self.worker_index
        self.provider.change_status(self.task, TaskStatus.InProgress)

    def download(self):
        if not self.task.debug:
            folder = self.storage.download(task=self.id)
        else:
            folder = os.getcwd()

        libraries = self.library_provider.dag(self.task.dag)
        was_installation = self.storage.import_folder(folder, libraries)
        if was_installation and not self.task.debug:
            if self.repeat_count > 0:
                try:
                    queue = f'{self.hostname}_{self.docker_img}_' \
                        f'{os.getenv("WORKER_INDEX")}'
                    self.logger.warning(traceback.format_exc(),
                                        ComponentType.Worker)
                    execute.apply_async((self.id, self.repeat_count - 1),
                                        queue=queue)
                except Exception:
                    pass
                finally:
                    sys.exit()
        os.chdir(folder)

    def create_executor(self):
        config = Config.from_yaml(self.dag.config)
        executor_type = config['executors'][self.task.executor]['type']

        assert Executor.is_registered(executor_type), \
            f'Executor {executor_type} was not found'

        additional_info = yaml_load(self.task.additional_info) \
            if self.task.additional_info else dict()
        self.executor = Executor.from_config(self.task.executor,
                                             config,
                                             additional_info)

    def execute(self):
        self.executor(self.task, self.dag)

        self.provider.change_status(self.task, TaskStatus.Success)

    def build(self):
        try:
            self.create_base()

            self.check_status()

            self.change_status()

            self.download()

            self.create_executor()

            self.execute()

        except Exception as e:
            if type(e) == ProgrammingError:
                Session.cleanup()

            step = self.executor.step.id if \
                (self.executor and self.executor.step) else None

            self.logger.error(traceback.format_exc(),
                              ComponentType.Worker,
                              self.hostname,
                              self.id,
                              step)
            self.provider.change_status(self.task, TaskStatus.Failed)
            raise e
        finally:
            sys.exit()


def execute_by_id(id: int, repeat_count=1):
    ex = ExecuteBuilder(id, repeat_count)
    ex.build()


@celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ["WORKER_INDEX"] = sender.split('@')[1]


@app.task
def execute(id: int, repeat_count: int = 1):
    execute_by_id(id, repeat_count)


@app.task
def kill(pid: int):
    os.system(f'kill -9 {pid}')


@app.task
def remove(path: str):
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def remove_from_all(path: str):
    provider = DockerProvider()
    dockers = provider.get_online()
    for docker in dockers:
        queue = f'{docker.computer}_{docker.name or "default"}_supervisor'
        remove.apply((path,), queue=queue)


def remove_model(project_name: str, model_name: str):
    path = join(MODEL_FOLDER, project_name, model_name + '.pth')
    remove_from_all(path)


def remove_task(id: int):
    path = join(TASK_FOLDER, str(id))
    remove_from_all(path)


def remove_dag(id: int):
    tasks = TaskProvider().by_dag(id)
    for task in tasks:
        remove_task(task.id)


def stop(task: Task):
    assert task.dag_rel, 'Dag is not in the task'
    logger = create_logger()
    provider = TaskProvider()
    if task.status > TaskStatus.InProgress.value:
        return task.status

    try:
        app.control.revoke(task.celery_id, terminate=True)
    except Exception as e:
        if type(e) == ProgrammingError:
            Session.cleanup()
        logger.error(traceback.format_exc(), ComponentType.API)
    finally:
        if task.pid:
            queue = f'{task.computer_assigned}_' \
                f'{task.dag_rel.docker_img or "default"}_supervisor'
            kill.apply_async((task.pid,), queue=queue)
        provider.change_status(task, TaskStatus.Stopped)

    return task.status


if __name__ == '__main__':
    execute(81)
    # from task.tasks import execute
    # execute.delay(42)
