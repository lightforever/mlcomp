import time

from mlcomp.worker.executors import Executor


@Executor.register
class Progress(Executor):
    def work(self):
        items = list(range(1000))
        for i in self.tqdm(items, interval=1):
            time.sleep(0.01)
