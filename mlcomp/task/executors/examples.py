from mlcomp.task.executors.base import Executor

@Executor.register
class Dummy(Executor):
    def work(self):
        pass


@Executor.register
class StepExample(Executor):
    def work(self):
        self.step.start(1, 'step 1.1')
        self.step.start(1, 'step 1.2')
        self.step.start(2, 'step 1.2.1')
        self.step.start(3, 'step 1.2.1.1')
        self.step.start(3, 'step 1.2.1.2')
        self.step.start(2, 'step 1.2.2')

        self.step.end(2)
        self.step.end(0)