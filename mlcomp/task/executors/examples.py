from .base import Executor

@Executor.register
class Dummy(Executor):
    def work(self):
        pass