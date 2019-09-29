import os
import shutil
import socket
import time
import traceback
import sys
from os.path import join, dirname, abspath

from sqlalchemy.orm import joinedload
from celery.signals import celeryd_after_setup
from celery import states

from mlcomp import MODEL_FOLDER, TASK_FOLDER, DOCKER_IMG
from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType, TaskStatus
from mlcomp.db.models import Task, Dag
from mlcomp.db.providers import TaskProvider, \
    DagLibraryProvider, \
    DockerProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.misc import set_global_seed
from mlcomp.worker.app import app
from mlcomp.worker.executors import Executor
from mlcomp.worker.storage import Storage
from mlcomp.utils.config import Config


class ExecuteBuilder:
    def __init__(self, id: int, repeat_count: int = 1, exit=True):
        self.session = Session.create_session(key='ExecuteBuilder')
        self.id = id
        self.repeat_count = repeat_count
        self.logger = create_logger(self.session, 'ExecuteBuilder')
        self.exit = exit

        self.provider = None
        self.library_provider = None
        self.storage = None
        self.task = None
        self.dag = None
        self.executor = None
        self.hostname = None
        self.docker_img = None
        self.worker_index = None
        self.queue_personal = None
        self.config = None
        self.executor_type = None

    def info(self, msg: str, step=None):
        self.logger.info(msg, ComponentType.Worker, self.hostname, self.id,
                         step)

    def error(self, msg: str, step=None):
        self.logger.error(msg, ComponentType.Worker, self.hostname, self.id,
                          step)

    def warning(self, msg: str, step=None):
        self.logger.warning(msg, ComponentType.Worker, self.hostname, self.id,
                            step)

    def debug(self, msg: str, step=None):
        self.logger.debug(msg, ComponentType.Worker, self.hostname, self.id,
                          step)

    def create_base(self):
        self.info('create_base')

        if app.current_task:
            app.current_task.update_state(state=states.SUCCESS)
            app.control.revoke(app.current_task.request.id, terminate=True)

        self.provider = TaskProvider(self.session)
        self.library_provider = DagLibraryProvider(self.session)
        self.storage = Storage(self.session)

        self.task = self.provider.by_id(
            self.id, joinedload(Task.dag_rel, innerjoin=True)
        )
        if not self.task:
            raise Exception(f'task with id = {self.id} is not found')

        self.dag = self.task.dag_rel
        self.executor = None
        self.hostname = socket.gethostname()

        self.docker_img = DOCKER_IMG
        self.worker_index = os.getenv('WORKER_INDEX', -1)

        self.queue_personal = f'{self.hostname}_{self.docker_img}_' \
                              f'{self.worker_index}'

        self.config = Config.from_yaml(self.dag.config)

        set_global_seed(self.config['info'].get('seed', 0))

        self.executor_type = self.config['executors'][self.task.executor][
            'type']

        executor = self.config['executors'][self.task.executor]
        env = {
            'MKL_NUM_THREADS': 1,
            'OMP_NUM_THREADS': 1
        }
        env.update(executor.get('env', {}))

        for k, v in env.items():
            os.environ[k] = str(v)
            self.info(f'Set env. {k} = {v}')

    def check_status(self):
        self.info('check_status')

        assert self.dag is not None, 'You must fetch task with dag_rel'

        if self.task.status >= TaskStatus.InProgress.value:
            msg = f'Task = {self.task.id}. Status = {self.task.status}, ' \
                  f'before the execute_by_id invocation.'
            if app.current_task:
                msg += f' Request Id = {app.current_task.request.id}'
            self.error(msg)
            return True

    def change_status(self):
        self.info('change_status')

        self.task.computer_assigned = self.hostname
        self.task.pid = os.getpid()
        self.task.worker_index = self.worker_index
        self.task.docker_assigned = self.docker_img
        self.provider.change_status(self.task, TaskStatus.InProgress)

    def download(self):
        self.info('download')

        if not self.task.debug:
            folder = self.storage.download(task=self.id)
        else:
            folder = os.getcwd()

        os.chdir(folder)

        libraries = self.library_provider.dag(self.task.dag)
        executor_type = self.executor_type

        mlcomp_executors_folder = join(dirname(abspath(__file__)), 'executors')
        mlcomp_base_folder = os.path.abspath(join(mlcomp_executors_folder,
                                                  '../../../'))

        imported, was_installation = self.storage.import_executor(
            mlcomp_executors_folder,
            mlcomp_base_folder,
            executor_type)

        if not imported:
            imported, was_installation = self.storage.import_executor(
                folder,
                folder,
                executor_type,
                libraries)

            if not imported:
                raise Exception(f'Executor = {executor_type} not found')

        if was_installation and not self.task.debug:
            if self.repeat_count > 0:
                try:
                    self.warning(traceback.format_exc())
                    self.task.status = TaskStatus.Queued.value
                    self.provider.commit()

                    execute.apply_async(
                        (self.id, self.repeat_count - 1),
                        queue=self.queue_personal,
                        retry=False
                    )
                except Exception:
                    pass
                finally:
                    sys.exit()

        assert Executor.is_registered(executor_type), \
            f'Executor {executor_type} was not found'

    def create_executor(self):
        self.info('create_executor')

        os.environ['CUDA_VISIBLE_DEVICES'] = self.task.gpu_assigned or ''

        additional_info = yaml_load(self.task.additional_info) \
            if self.task.additional_info else dict()
        self.executor = Executor.from_config(
            executor=self.task.executor, config=self.config,
            additional_info=additional_info,
            session=self.session, logger=self.logger
        )

    def execute(self):
        self.info('execute start')

        res = self.executor(task=self.task, task_provider=self.provider,
                            dag=self.dag)
        self.info('execute executor finished')

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

                time.sleep(3)

                self.executor.info(f'sending {(self.id, self.repeat_count)} '
                                   f'to {self.queue_personal}')

                self.task.status = TaskStatus.Queued.value
                self.provider.commit()

                execute.apply_async(
                    (self.id, self.repeat_count), queue=self.queue_personal,
                    retry=False
                )
                return

        self.executor.step.finish()
        self.provider.change_status(self.task, TaskStatus.Success)

        self.info('execute end')

    def build(self):
        try:
            self.create_base()

            bad_status = self.check_status()
            if bad_status:
                return

            self.change_status()

            self.download()

            self.create_executor()

            self.execute()

        except Exception as e:
            step = self.executor.step.id if \
                (self.executor and self.executor.step) else None

            if Session.sqlalchemy_error(e):
                Session.cleanup(key='ExecuteBuilder')
                self.session = Session.create_session(key='ExecuteBuilder')
                self.logger.session = create_logger(self.session,
                                                    'ExecuteBuilder')

            self.error(traceback.format_exc(), step)
            if self.task.status <= TaskStatus.InProgress.value:
                self.provider.change_status(self.task, TaskStatus.Failed)
            raise e
        finally:
            if app.current_task:
                app.close()

            if self.exit:
                # noinspection PyProtectedMember
                os._exit(0)


def execute_by_id(id: int, repeat_count=1, exit=True):
    ex = ExecuteBuilder(id, repeat_count=repeat_count, exit=exit)
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
        remove.apply((path,), queue=queue)


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


def stop(logger, session: Session, task: Task, dag: Dag):
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
        if Session.sqlalchemy_error(e):
            try:
                logger.error(traceback.format_exc(), ComponentType.API)
            except Exception:
                pass
            raise
        logger.error(traceback.format_exc(), ComponentType.API)
    finally:
        if task.pid:
            queue = f'{task.computer_assigned}_' \
                    f'{dag.docker_img or "default"}_supervisor'
            kill.apply_async((task.pid,), queue=queue, retry=False)
        provider.change_status(task, status)

    return task.status


if __name__ == '__main__':
    execute(81)
    # from task.tasks import execute
    # execute.delay(42)
