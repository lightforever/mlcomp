import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from mlcomp.utils.config import Config
from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.equation import Equation


@Executor.register
class Infer(Equation, ABC):
    def __init__(
            self,
            *,
            equations: dict,
            targets: List[str] = ('\'y\'',),
            name: str = '\'infer\'',
            max_count=None,
            test: bool = False,
            prepare_submit: bool = False,
            layout: str = None,
            suffix: str = 'valid',
            plot_count: int = 0,
            **kwargs
    ):
        super().__init__(equations, targets, name)

        self.max_count = self.solve(max_count)
        self.test = self.solve(test)
        self.prepare_submit = self.solve(prepare_submit)
        self.layout = self.solve(layout)
        self.suffix = self.solve(suffix)
        self.plot_count = self.solve(plot_count)

    def plot(self, res):
        pass

    @abstractmethod
    def submit(self, res, folder):
        pass

    def work(self):
        res = super().work()
        folder = 'data/pred'
        os.makedirs(folder, exist_ok=True)

        np.save(f'{folder}/{self.name}_{self.suffix}', res['y'])

        if self.test and self.prepare_submit:
            folder = 'data/submissions'
            os.makedirs(folder, exist_ok=True)
            self.submit(res, folder)

        if self.layout:
            self.plot(res)
