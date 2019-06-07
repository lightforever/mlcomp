from .top_classify import TopClassifyCallback
from mlcomp.task.executors import Executor
from mlcomp.task.executors.catalyst import Catalyst
from mlcomp.task.executors.catalyst.f1 import F1Callback
from mlcomp.task.executors.catalyst.precision_recall import PrecisionRecallCallback

@Executor.register
class Segmenter(Catalyst):
    def callbacks(self):
        result = [self]
        for items, cls in [
            [self.report.precision_recall, PrecisionRecallCallback],
            [self.report.f1, F1Callback],
            [self.report.img_classify, TopClassifyCallback],
        ]:
            for item in items:
                result.append(cls(self.experiment, self.task, self.dag, item))
        return result