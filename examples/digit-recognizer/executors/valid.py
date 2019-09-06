import numpy as np
from sklearn.metrics import accuracy_score
import cv2

from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.valid import Valid
from mlcomp.worker.reports.classification import ClassificationReportBuilder

from dataset import MnistDataset


@Executor.register
class ValidMnist(Valid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x = MnistDataset(
            file='data/train.csv',
            fold_csv='data/fold.csv',
            is_test=True,
            max_count=self.max_count
        )

    def score(self, res):
        return accuracy_score(self.x.y, res.argmax(axis=1))

    def plot(self, res, score):
        imgs = [
            cv2.cvtColor(
                (row['features'][0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
            ) for row in self.x
        ]

        attrs = [
            {
                'attr1': p.argmax(),
                'attr2': t
            } for p, t in zip(res, self.x.y)
        ]

        builder = ClassificationReportBuilder(
            session=self.session,
            layout=self.layout,
            preds=res,
            targets=self.x.y,
            task=self.task,
            imgs=imgs,
            scores={'accuracy': score},
            name=self.name,
            attrs=attrs
        )
        builder.build()
