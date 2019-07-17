from sklearn.metrics import accuracy_score

from mlcomp.utils.config import Config
from mlcomp.worker.executors import Executor
from mlcomp.worker.interfaces import Interface

from dataset import MnistDataset


@Executor.register
class Valid(Executor):
    __syn__ = 'valid'

    def __init__(self,
                 model: Interface):
        self.model = model
        self.dataset = MnistDataset(file='data/train.csv',
                                    fold_csv='data/fold.csv',
                                    is_test=True
                                    )

    def work(self):
        input = {
            'dataset': self.dataset
        }
        res = self.model(input)['prob'].argmax(axis=1)
        score = accuracy_score(self.dataset.y, res)
        self.task.score = score

    @classmethod
    def _from_config(cls,
                     executor: dict,
                     config: Config,
                     additional_info: dict):
        model = Interface.from_config(executor['slot'])
        return cls(model)
