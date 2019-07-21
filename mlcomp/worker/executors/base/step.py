import json

from mlcomp.db.enums import StepStatus, ComponentType
from mlcomp.db.models import Task, Step
from mlcomp.db.providers import LogProvider, StepProvider, TaskProvider
from mlcomp.utils.logging import create_logger
from mlcomp.utils.misc import now


class StepWrap:
    def __init__(self, task: Task):
        self.log_provider = LogProvider()
        self.step_provider = StepProvider()
        self.task_provider = TaskProvider()
        self.task = task
        self.children = []
        self.step = None
        self.logger = create_logger()

    @property
    def id(self):
        return self.step.id

    def __enter__(self):
        self.step = self.start(0, 'main')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end(0, failed=exc_type is Exception)

    def _finish(self, metrics: dict = None, failed: bool = False):
        if len(self.children) == 0:
            return
        step = self.children.pop()
        step.metrics = json.dumps(metrics) if metrics is not None else None
        step.status = StepStatus.Failed.value \
            if failed else StepStatus.Successed.value
        step.finished = now()
        self.step_provider.update()
        self.step = step

        self.info('End of the step')

    def start(self, level: int, name: str):
        if self.step is not None:
            diff = level - self.step.level
            assert level > 0, 'level must be positive'
            assert diff <= 1, \
                f'Level {level} can not be started after {self.step.level}'

            if diff <= 0:
                for _ in range(abs(diff) + 1):
                    self._finish()

        step = Step(level=level,
                    name=name,
                    started=now(),
                    status=StepStatus.InProgress.value,
                    task=self.task.id
                    )
        self.step_provider.add(step)
        self.children.append(step)
        self.step = step
        if level == 1:
            self.task.current_step = self.task.current_step + 1 \
                if self.task.current_step is not None else 1
            self.task_provider.session.commit()

        self.info('Begin of the step')

        return step

    def end(self, level: int, metrics: dict = None, failed=False):
        diff = level - self.step.level
        assert diff <= 0, 'you can end only the same step or lower'
        for i in range(abs(diff) + 1):
            if i == 0:
                self._finish(metrics, failed=failed)
            else:
                self._finish(None, failed=failed)

    def debug(self, message: str):
        self.logger.debug(message,
                          ComponentType.Worker,
                          self.task.computer_assigned,
                          self.task.id,
                          self.step.id)

    def info(self, message: str):
        self.logger.info(message,
                         ComponentType.Worker,
                         self.task.computer_assigned,
                         self.task.id,
                         self.step.id)

    def warning(self, message: str):
        self.logger.warning(message,
                            ComponentType.Worker,
                            self.task.computer_assigned,
                            self.task.id,
                            self.step.id)

    def error(self, message: str):
        self.logger.error(message,
                          ComponentType.Worker,
                          self.task.computer_assigned,
                          self.task.id,
                          self.step.id)


__all__ = ['StepWrap']