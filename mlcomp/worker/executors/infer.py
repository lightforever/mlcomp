import os
from abc import ABC, abstractmethod

import numpy as np

from mlcomp.utils.config import Config
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.equation import Equation

from dataset import MnistDataset


@Executor.register
class Infer(Equation, ABC):
    def __init__(
        self,
        *,
        equations: dict,
        target: str = '\'y\'',
        name: str = '\'infer\'',
        max_count=None,
        test: bool = False,
        prepare_submit: bool = False,
        layout: str = None,
        suffix: str = 'valid',
        **kwargs
    ):
        super().__init__(equations, target, name)

        self.max_count = self.solve(max_count)
        self.test = self.solve(test)
        self.prepare_submit = self.solve(prepare_submit)
        self.layout = self.solve(layout)
        self.suffix = self.solve(suffix)

        if self.test:
            self.x = MnistDataset(file='data/test.csv')
        else:
            self.x = MnistDataset(
                file='data/train.csv', fold_csv='data/fold.csv', is_test=True
            )

    def plot(self, res):
        pass

    @abstractmethod
    def submit(self, res, folder):
        pass

    def work(self):
        res = super().work()['res']
        np.save(f'data/{self.name}_{self.suffix}', res)

        if self.test and self.prepare_submit:
            folder = 'data/submissions'
            os.makedirs(folder, exist_ok=True)
            self.submit(res, folder)

        if self.layout:
            self.plot(res)

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
