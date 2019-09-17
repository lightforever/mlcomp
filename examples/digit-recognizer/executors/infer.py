import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from mlcomp.worker.executors import Executor

from mlcomp.worker.executors.infer import Infer
from mlcomp.worker.reports.classification import ClassificationReportBuilder

from dataset import MnistDataset
from experiment import Experiment


@Executor.register
class InferMnist(Infer):
    def __init__(self, **kwargs):
        cache_names = ['y']
        super().__init__(cache_names=cache_names, layout='img_classify',
                         **kwargs)

        if self.test:
            self.x_source = MnistDataset(
                file='data/test.csv',
                transforms=Experiment.get_test_transforms(),
                max_count=self.max_count,
            )
        else:
            self.x_source = MnistDataset(
                file='data/train.csv',
                fold_csv='data/fold.csv',
                is_test=True,
                transforms=Experiment.get_test_transforms(),
                max_count=self.max_count
            )

        self.builder = None
        self.x = None
        self.res = []
        self.submit_res = []

    def create_base(self):
        self.builder = ClassificationReportBuilder(
            session=self.session,
            task=self.task,
            layout=self.layout,
            name=self.name,
            plot_count=self.plot_count
        )

    def count(self):
        return len(self.x_source)

    def adjust_part(self, part):
        self.x = deepcopy(self.x_source)
        self.x.x = self.x.x[part[0]:part[1]]
        if not self.test:
            self.x.y = self.x.y[part[0]:part[1]]

    def save(self, preds, folder: str):
        self.res.extend(preds)

    def save_final(self, folder):
        pickle.dump(np.array(self.res),
                    open(f'{folder}/{self.model_name}_{self.suffix}.p', 'wb'))

    def submit(self, preds):
        argmax = preds.argmax(axis=1)
        self.submit_res.extend(
            [{'ImageId': len(self.submit_res) + i + 1, 'Label': p} for i, p in
             enumerate(argmax)])

    def submit_final(self, folder):
        pd.DataFrame(self.submit_res). \
            to_csv(f'{folder}/{self.model_name}_{self.suffix}.csv',
                   index=False)

    def _plot_main(self, preds):
        imgs = [
            ((row['features'][0] * 0.229 + 0.485) * 255).astype(np.uint8)
            for row in self.x
        ]
        attrs = [
            {
                'attr1': p.argmax()
            } for p in preds
        ]

        self.builder.process_pred(
            imgs=imgs,
            preds=preds,
            attrs=attrs
        )

    def plot(self, preds):
        self._plot_main(preds)
