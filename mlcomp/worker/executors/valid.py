from abc import ABC, abstractmethod
from typing import List

from mlcomp.db.providers import ModelProvider
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.equation import Equation


@Executor.register
class Valid(Equation, ABC):
    def __init__(
        self,
        equations: dict,
        max_count=None,
        layout=None,
        model_id=None,
        fold_number: int = 0,
        plot_count: int = 0,
        **kwargs
    ):
        super().__init__(equations, targets, name)

        self.max_count = self.solve(max_count)
        self.layout = self.solve(layout)
        self.fold_number = self.solve(fold_number)
        self.plot_count = self.solve(plot_count)
        self.model_id = model_id

    @abstractmethod
    def score(self, res):
        pass

    def plot(self, res, scores):
        pass

    def work(self):
        res = super().work()
        score, scores = self.score(res)
        score = float(score)

        self.task.score = score
        self.task_provider.update()

        if self.model_id:
            provider = ModelProvider(self.session)
            model = provider.by_id(self.model_id)
            model.score_local = score
            provider.commit()

        if self.layout:
            self.plot(res, scores)
