from abc import ABC, abstractmethod
import time

from mlcomp.utils.io import yaml_load, yaml_dump

from mlcomp import FILE_SYNC_INTERVAL
from mlcomp.db.core import Session
from mlcomp.db.models import Task, Dag
from mlcomp.utils.config import Config
from mlcomp.db.providers import TaskProvider, TaskSyncedProvider
from mlcomp.utils.misc import to_snake
from mlcomp.worker.executors.base.step import StepWrap


class Executor(ABC):
    _child = dict()

    session = None
    task_provider = None
    logger = None
    step = None

    def __init__(self, **kwargs):
        pass

    def debug(self, message: str):
        if self.step:
            self.step.debug(message)
        else:
            print(message)

    def info(self, message: str):
        if self.step:
            self.step.info(message)
        else:
            print(message)

    def warning(self, message: str):
        if self.step:
            self.step.warning(message)
        else:
            print(message)

    def error(self, message: str):
        if self.step:
            self.step.error(message)
        else:
            print(message)

    def add_child_process(self, pid: int):
        additional_info = yaml_load(self.task.additional_info)
        additional_info['child_processes'] = additional_info.get(
            'child_processes', []) + [pid]
        self.task.additional_info = yaml_dump(additional_info)
        self.task_provider.update()

    def __call__(
        self, *, task: Task, task_provider: TaskProvider, dag: Dag
    ) -> dict:
        assert dag is not None, 'You must fetch task with dag_rel'

        self.task_provider = task_provider
        self.task = task
        self.dag = dag
        self.step = StepWrap(self.session, self.logger, task, task_provider)
        self.step.enter()

        if not task.debug and FILE_SYNC_INTERVAL:
            self.wait_data_sync()
        res = self.work()
        self.step.task_provider.commit()
        return res

    @abstractmethod
    def work(self) -> dict:
        pass

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        return cls(**executor)

    @staticmethod
    def from_config(
        *, executor: str, config: Config, additional_info: dict,
        session: Session, logger
    ) -> 'Executor':
        if executor not in config['executors']:
            raise ModuleNotFoundError(
                f'Executor {executor} '
                f'has not been found'
            )

        executor = config['executors'][executor]
        child_class = Executor._child[executor['type']]
        # noinspection PyProtectedMember
        res = child_class._from_config(executor, config, additional_info)
        res.session = session
        res.logger = logger
        return res

    @staticmethod
    def register(cls):
        Executor._child[cls.__name__] = cls
        Executor._child[cls.__name__.lower()] = cls
        Executor._child[to_snake(cls.__name__)] = cls
        return cls

    @staticmethod
    def is_registered(cls: str):
        return cls in Executor._child

    def wait_data_sync(self):
        self.info(f'Start data sync')

        while True:
            provider = TaskSyncedProvider(self.session)
            provider.commit()

            wait = False
            for computer, project, tasks in provider.for_computer(
                self.task.computer_assigned
            ):
                if project.id == self.dag.project:
                    wait = True

            if wait:
                time.sleep(1)
            else:
                break

        self.info(f'Finish data sync')

    @staticmethod
    def is_trainable(type: str):
        variants = ['Catalyst']
        return type in (variants + [v.lower() for v in variants])


__all__ = ['Executor']
