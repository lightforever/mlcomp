import os
from abc import ABC, abstractmethod

from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.base.equation import Equation


@Executor.register
class PrepareSubmit(Equation, ABC):
    def __init__(
            self,
            layout: str = None,
            plot_count: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.layout = layout
        self.plot_count = plot_count

    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def plot(self, preds):
        pass

    def key(self):
        return 'y'

    @abstractmethod
    def create_base(self):
        pass

    @abstractmethod
    def submit(self, preds):
        pass

    @abstractmethod
    def submit_final(self, folder):
        pass

    @abstractmethod
    def adjust_part(self, part):
        pass

    def work(self):
        submit_folder = 'data/submissions'
        os.makedirs(submit_folder, exist_ok=True)

        self.create_base()
        parts = self.generate_parts(self.count())

        for preds in self.solve(self.key(), parts):
            self.submit(preds)

            if self.layout:
                self.plot(preds)

        self.submit_final(submit_folder)
