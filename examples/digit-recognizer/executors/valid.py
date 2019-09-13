import numpy as np
import cv2

from mlcomp.worker.executors import Executor
from mlcomp.worker.executors.valid import Valid
from mlcomp.worker.reports.classification import ClassificationReportBuilder

from dataset import MnistDataset
from experiment import Experiment


@Executor.register
class ValidMnist(Valid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x = MnistDataset(
            file='data/train.csv',
            fold_csv='data/fold.csv',
            is_test=True,
            max_count=self.max_count,
            transforms=Experiment.get_test_transforms()
        )

    def score(self):
        # noinspection PyUnresolvedReferences
        res = self.solve(self.y)
        scores = res[np.arange(len(self.x)), self.x.y]
        return np.mean(scores), scores

    def plot(self, scores):
        res = self.solve(self.y)
        imgs = [
            cv2.cvtColor(
                ((row['features'][0] * 0.229 + 0.485) * 255).astype(np.uint8),
                cv2.COLOR_GRAY2BGR
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
            scores={'accuracy': scores},
            name=self.name,
            attrs=attrs,
            plot_count=self.plot_count
        )
        builder.build()
