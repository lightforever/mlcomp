import os
import shutil
import socket
import time
import traceback
import sys
from os.path import join

from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import joinedload
from celery.signals import celeryd_after_setup

from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus
from mlcomp.db.models import Task, Dag
from mlcomp.db.providers import TaskProvider, \
    DagLibraryProvider, \
    DockerProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.worker.app import app
from mlcomp.worker.executors import Executor
from mlcomp.worker.storage import Storage
from mlcomp.utils.config import Config
from mlcomp.utils.settings import MODEL_FOLDER, TASK_FOLDER


class ExecuteBuilder:
    def __init__(self, session: Session, id: int, repeat_count: int = 1):
        self.id = id
        self.repeat_count = repeat_count
        self.logger = create_logger(session)

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
        self.queue_personal = None
        self.session = session

    def create_base(self):
        self.provider = TaskProvider(self.session)
        self.library_provider = DagLibraryProvider(self.session)
        self.storage = Storage(self.session)

        self.task = self.provider.by_id(
            self.id, joinedload(Task.dag_rel, innerjoin=True)
        )
        self.dag = self.task.dag_rel
        self.executor = None
        self.hostname = socket.gethostname()

        self.docker_img = os.getenv('DOCKER_IMG', 'default')
        self.worker_index = int(os.getenv('WORKER_INDEX', -1))

        self.queue_personal = f'{self.hostname}_{self.docker_img}_' \
                              f'{os.getenv("WORKER_INDEX")}'

    def check_status(self):
        assert self.dag is not None, 'You must fetch task with dag_rel'

        if self.task.status > TaskStatus.InProgress.value:

            msg = f'Task = {self.task.id}. Status = {self.task.status}, ' \
                  f'before the execute_by_id invocation'
            self.logger.error(msg, ComponentType.Worker)
            return

    def change_status(self):
        self.task.computer_assigned = self.hostname
        self.task.pid = os.getpid()
        self.task.worker_index = self.worker_index
        self.task.docker_assigned = self.docker_img
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
                    self.logger.warning(
                        traceback.format_exc(), ComponentType.Worker
                    )
                    execute.apply_async(
                        (self.id, self.repeat_count - 1),
                        queue=self.queue_personal
                    )
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
        self.executor = Executor.from_config(
            self.task.executor, config, additional_info
        )

    def execute(self):
        res = self.executor(self.task, self.dag)
        res = res or {}
        self.task.result = yaml_dump(res)
        self.provider.commit()

        if 'stage' in res and 'stages' in res:
            index = res['stages'].index(res['stage'])
            if index < len(res['stages']) - 1:
                self.executor.info(
                    f'stage = {res["stage"]} done. '
                    f'Go to the stage = '
                    f'{res["stages"][index + 1]}'
                )

                time.sleep(5)
                execute.apply_async(
                    (self.id, self.repeat_count), queue=self.queue_personal
                )
                return

        if self.task.current_step is None:
            self.task.current_step = self.task.steps

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

            self.logger.error(
                traceback.format_exc(), ComponentType.Worker, self.hostname,
                self.id, step
            )
            self.provider.change_status(self.task, TaskStatus.Failed)
            raise e
        finally:
            sys.exit()


def execute_by_id(id: int, repeat_count=1):
    session = Session.create_session(key='tasks.execute_by_id')
    ex = ExecuteBuilder(session, id, repeat_count)
    ex.build()


@celeryd_after_setup.connect
def capture_worker_name(sender, instance, **kwargs):
    os.environ['WORKER_INDEX'] = sender.split('_')[-1]


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


def remove_from_all(session: Session, path: str):
    provider = DockerProvider(session)
    dockers = provider.get_online()
    for docker in dockers:
        queue = f'{docker.computer}_{docker.name or "default"}_supervisor'
        remove.apply((path, ), queue=queue)


def remove_model(session: Session, project_name: str, model_name: str):
    path = join(MODEL_FOLDER, project_name, model_name + '.pth')
    remove_from_all(session, path)


def remove_task(session: Session, id: int):
    path = join(TASK_FOLDER, str(id))
    remove_from_all(session, path)


def remove_dag(session: Session, id: int):
    tasks = TaskProvider(session).by_dag(id)
    for task in tasks:
        remove_task(session, task.id)


def stop(session: Session, task: Task, dag: Dag):
    logger = create_logger(session)
    provider = TaskProvider(session)
    if task.status > TaskStatus.InProgress.value:
        return task.status

    status = TaskStatus.Stopped
    try:
        if task.status != TaskStatus.NotRan.value:
            app.control.revoke(task.celery_id, terminate=True)
        else:
            status = TaskStatus.Skipped
    except Exception as e:
        if type(e) == ProgrammingError:
            Session.cleanup()
        logger.error(traceback.format_exc(), ComponentType.API)
    finally:
        if task.pid:
            queue = f'{task.computer_assigned}_' \
                    f'{dag.docker_img or "default"}_supervisor'
            kill.apply_async((task.pid, ), queue=queue)
        provider.change_status(task, status)

    return task.status


if __name__ == '__main__':
    execute(81)
    # from task.tasks import execute
    # execute.delay(42)
