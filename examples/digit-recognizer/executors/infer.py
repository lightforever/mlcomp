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
        super().__init__(**kwargs)

        if self.test:
            self.x = MnistDataset(
                file='data/test.csv',
                transforms=Experiment.get_test_transforms(),
                max_count=self.max_count,
            )
        else:
            self.x = MnistDataset(
                file='data/train.csv',
                fold_csv='data/fold.csv',
                is_test=True,
                transforms=Experiment.get_test_transforms(),
                max_count=self.max_count
            )

    def submit(self, res, folder):
        res = res['y']
        argmax = res.argmax(axis=1)
        pd.DataFrame(
            {
                'ImageId': np.arange(1,
                                     len(argmax) + 1),
                'Label': argmax
            }
        ).to_csv(f'{folder}/{self.name}.csv', index=False)

    def plot(self, res):
        res = res['y']
        imgs = [
            ((row['features'][0] * 0.229 + 0.485) * 255).astype(np.uint8)
            for row in self.x
        ]
        attrs = [
            {
                'attr1': p.argmax()
            } for p in res
        ]
        builder = ClassificationReportBuilder(
            session=self.session,
            task=self.task,
            layout=self.layout,
            preds=res,
            targets=None if self.test else self.x.y,
            imgs=imgs,
            name=self.name,
            attrs=attrs,
            plot_count=self.plot_count
        )
        builder.build()
