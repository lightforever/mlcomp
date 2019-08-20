from mlcomp.worker.executors import Executor


@Executor.register
class Step(Executor):
    def work(self):
        self.step.start(1, 'step 1')

        self.step.start(1, 'step 2')

        self.step.start(2, 'step 2.1')

        self.step.start(3, 'step 2.1.1')

        self.step.start(3, 'step 2.1.2')

        self.step.start(2, 'step 2.2')

        self.step.end(0)
