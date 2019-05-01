from .base import Executor

@Executor.register
class Dummy(Executor):
    def work(self):
        pass


@Executor.register
class StepExample(Executor):
    def work(self):
        self.step.start(1, 'step 1')
        self.step.start(1, 'step 1.1')
        self.step.start(2, 'step 1.1.1')
        self.step.start(3, 'step 1.1.1')

        self.step.end(3)
        self.step.end(0)