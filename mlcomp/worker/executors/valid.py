from abc import ABC, abstractmethod

from numpy import isnan

from mlcomp.db.providers import ModelProvider
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.equation import Equation


@Executor.register
class Valid(Equation, ABC):
    def __init__(
            self,
            layout=None,
            fold_number: int = 0,
            plot_count: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.layout = layout
        self.fold_number = fold_number
        self.plot_count = plot_count

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def plot(self, preds, score):
        pass

    @abstractmethod
    def plot_final(self, score):
        pass

    @abstractmethod
    def score(self, preds):
        pass

    @abstractmethod
    def score_final(self):
        pass

    @abstractmethod
    def create_base(self):
        pass

    @abstractmethod
    def adjust_part(self, part):
        pass

    def key(self):
        return 'y'

    def work(self):
        self.create_base()
        parts = self.generate_parts(self.count())

        for preds in self.solve(self.key(), parts):
            score = self.score(preds)
            if self.layout and self.plot_count > 0:
                self.plot(preds, score)

        score = self.score_final()
        if isnan(score):
            score = -1

        if self.layout:
            self.plot_final(score)

        self.task.score = score
        self.task_provider.update()

        if self.model_id:
            provider = ModelProvider(self.session)
            model = provider.by_id(self.model_id)
            model.score_local = score
            provider.commit()
