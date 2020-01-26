from mlcomp.db.core import Session
from mlcomp.db.enums import ComponentType
from mlcomp.db.models import Task, Step
from mlcomp.db.providers import LogProvider, StepProvider, TaskProvider
from mlcomp.utils.misc import now


class StepWrap:
    def __init__(
            self, session: Session, logger, logger_db, task: Task,
            task_provider: TaskProvider
    ):
        self.log_provider = LogProvider(session)
        self.step_provider = StepProvider(session)
        self.task_provider = task_provider
        self.task = task
        self.children = []
        self.step = None
        self.logger = logger
        self.logger_db = logger_db

    @property
    def id(self):
        return self.step.id

    def enter(self):
        task = self.task if not self.task.parent else self.task_provider.by_id(
            self.task.parent
        )
        self.children = self.step_provider.unfinished(task.id)
        if len(self.children) == 0:
            self.step = self.start(0, 'main', 0)
        else:
            self.step = self.children[-1]

    def _finish(self):
        if len(self.children) == 0:
            return
        step = self.children.pop()
        step.finished = now()
        self.step_provider.update()
        self.step = self.children[-1] if len(self.children) > 0 else step

        self.debug('End of the step')

    def finish(self):
        while len(self.children) > 0:
            self._finish()

    def start(self, level: int, name: str = None, index: int = None):
        task = self.task if not self.task.parent else self.task_provider.by_id(
            self.task.parent
        )

        if index is None and task.current_step:
            parts = task.current_step.split('.')
            if len(parts) >= level:
                index = int(parts[level - 1])

        if self.step and index == self.step.index and self.step.level == level:
            return

        if self.step is not None:
            diff = level - self.step.level
            assert level > 0, 'level must be positive'
            assert diff <= 1, \
                f'Level {level} can not be started after {self.step.level}'

            if diff <= 0:
                for _ in range(abs(diff) + 1):
                    self._finish()

        step = Step(
            level=level,
            name=name or '',
            started=now(),
            task=task.id,
            index=index or 0
        )
        self.step_provider.add(step)
        self.children.append(step)
        self.step = step

        task.current_step = '.'.join(
            [
                str(c.index + 1)
                for c in self.children[1:]
            ]
        )
        self.task_provider.commit()

        self.debug('Begin of the step')

        return step

    def end(self, level: int):
        diff = level - self.step.level
        assert diff <= 0, 'you can end only the same step or lower'
        for i in range(abs(diff) + 1):
            self._finish()

    def debug(self, message: str, db: bool = False):
        logger = self.logger_db if db else self.logger
        logger.debug(
            message, ComponentType.Worker, self.task.computer_assigned,
            self.task.id, self.step.id
        )

    def info(self, message: str, db: bool = False):
        logger = self.logger_db if db else self.logger
        logger.info(
            message, ComponentType.Worker, self.task.computer_assigned,
            self.task.id, self.step.id
        )

    def warning(self, message: str, db: bool = False):
        logger = self.logger_db if db else self.logger
        logger.warning(
            message, ComponentType.Worker, self.task.computer_assigned,
            self.task.id, self.step.id
        )

    def error(self, message: str, db: bool = False):
        logger = self.logger_db if db else self.logger
        logger.error(
            message, ComponentType.Worker, self.task.computer_assigned,
            self.task.id, self.step.id
        )


__all__ = ['StepWrap']
