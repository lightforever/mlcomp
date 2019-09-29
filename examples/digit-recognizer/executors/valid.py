from copy import deepcopy

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
        cache_names = ['y']
        super().__init__(cache_names=cache_names, layout='img_classify',
                         **kwargs)

        self.x_source = MnistDataset(
            file='data/train.csv',
            fold_csv='data/fold.csv',
            is_test=True,
            max_count=self.max_count,
            transforms=Experiment.get_test_transforms()
        )
        self.builder = None
        self.x = None
        self.scores = []

    def create_base(self):
        self.builder = ClassificationReportBuilder(
            session=self.session,
            task=self.task,
            layout=self.layout,
            name=self.name,
            plot_count=self.plot_count
        )
        self.builder.create_base()

    def count(self):
        return len(self.x_source)

    def score(self, preds):
        # noinspection PyUnresolvedReferences
        res = (preds.argmax(axis=1) == self.x.y).astype(np.float)
        self.scores.extend(res)
        return res

    def score_final(self):
        return np.mean(self.scores)

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.x = self.x.x[part[0]:part[1]]
        self.x.y = self.x.y[part[0]:part[1]]

    def plot(self, preds, scores):
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
            } for p, t in zip(preds, self.x.y)
        ]

        self.builder.process_pred(
            imgs=imgs,
            preds=preds,
            targets=self.x.y,
            attrs=attrs,
            scores={'accuracy': scores}
        )

    def plot_final(self, score):
        self.builder.process_scores({'accuracy': score})
