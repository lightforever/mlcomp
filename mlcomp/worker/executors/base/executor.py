from abc import ABC, abstractmethod
import time

from tqdm import tqdm

from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp import FILE_SYNC_INTERVAL
from mlcomp.db.core import Session
from mlcomp.db.models import Task, Dag
from mlcomp.utils.config import Config
from mlcomp.db.providers import TaskProvider, TaskSyncedProvider
from mlcomp.utils.misc import to_snake
from mlcomp.worker.executors.base.step import StepWrap


class TqdmWrapper:
    def __init__(
            self, executor: 'Executor', iterable=None,
            desc: str = 'progress', interval: int = 10,
            **kwargs
    ):
        self.desc = desc
        self.iterable = iterable
        self.interval = interval
        self.executor = executor
        self.tqdm = tqdm(iterable=iterable, **kwargs)

    def refresh(self):
        executor = self.executor
        tqdm = self.tqdm

        executor.task.loader_name = self.desc
        executor.task.batch_index = tqdm.n
        executor.task.batch_total = tqdm.total
        executor.task.epoch_duration = time.time() - tqdm.start_t
        if tqdm.n > 0:
            frac = (tqdm.total - tqdm.n) / tqdm.n
            executor.task.epoch_time_remaining = \
                executor.task.epoch_duration * frac

        executor.task_provider.update()
        return time.time()

    def set_description(self, desc=None, refresh=True):
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
        self.desc = desc or ''
        if refresh:
            self.refresh()

    def __iter__(self):
        last_written = self.refresh()

        for item in self.tqdm:
            elapsed = time.time() - last_written
            if elapsed > self.interval:
                last_written = self.refresh()
            yield item
        self.refresh()


class Executor(ABC):
    _child = dict()

    session = None
    task_provider = None
    logger = None
    logger_db = None
    step = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def debug(self, message: str, db: bool = False):
        if self.step:
            self.step.debug(message, db=db)

    def info(self, message: str, db: bool = False):
        if self.step:
            self.step.info(message, db=db)

    def warning(self, message: str, db: bool = False):
        if self.step:
            self.step.warning(message, db=db)

    def error(self, message: str, db: bool = False):
        if self.step:
            self.step.error(message, db=db)

    def write(self, message: str):
        if message.strip() != '':
            self.info(message, db=True)

    def flush(self):
        pass

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
        self.step = StepWrap(self.session, self.logger, self.logger_db, task,
                             task_provider)
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
            session: Session, logger, logger_db
    ) -> 'Executor':
        if executor not in config['executors']:
            raise ModuleNotFoundError(
                f'Executor {executor} '
                f'has not been found'
            )

        executor = additional_info['executor']
        child_class = Executor._child[executor['type']]

        # noinspection PyProtectedMember
        res = child_class._from_config(executor, config, additional_info)
        res.session = session
        res.logger = logger
        res.logger_db = logger_db
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

    def tqdm(self, iterable=None, desc: str = 'progress', interval: int = 10,
             **kwargs):
        """
        tqdm wrapper. writes progress to Database
        Args:
            iterable: iterable for tqdm
            desc: name of the progress bar
            interval: interval of writing to Database
            **kwargs: tqdm additional arguments

        Returns: tqdm wrapper
        """
        return TqdmWrapper(
            executor=self, iterable=iterable,
            desc=desc, interval=interval,
            **kwargs)

    def dependent_results(self):
        tasks = self.task_provider.find_dependents(self.task.id)
        res = dict()
        for t in tasks:
            res[t.id] = yaml_load(t.result)
        return res


__all__ = ['Executor']
