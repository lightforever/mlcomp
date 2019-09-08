from abc import ABC, abstractmethod

from mlcomp.db.providers import ModelProvider
from mlcomp.utils.config import Config
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.equation import Equation


@Executor.register
class Valid(Equation, ABC):
    def __init__(
        self,
        equations: dict,
        target: str = '\'y\'',
        name: str = '\'valid\'',
        max_count=None,
        layout=None,
        model_id=None,
        fold_number=0,
        **kwargs
    ):
        super().__init__(equations, target, name)

        self.max_count = self.solve(max_count)
        self.layout = self.solve(layout)
        self.fold_number = self.solve(fold_number)
        self.model_id = model_id

    @abstractmethod
    def score(self, res):
        pass

    def plot(self, res, scores):
        pass

    def work(self):
        res = super().work()['res']
        score, scores = self.score(res)
        self.task.score = score
        self.task_provider.update()

        if self.model_id:
            provider = ModelProvider(self.session)
            model = provider.by_id(self.model_id)
            model.score_local = score
            provider.commit()

        if self.layout:
            self.plot(res, scores)

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        equations = cls.split(additional_info.get('equations', ''))
        kwargs = equations.copy()
        kwargs['equations'] = equations
        kwargs['model_id'] = additional_info.get('model_id')
        kwargs.update({k: Equation.encode(v) for k, v in executor.items()})
        return cls(**kwargs)
