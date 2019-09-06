import numpy as np
import pandas as pd

from mlcomp.worker.executors import Executor

from dataset import MnistDataset
from mlcomp.worker.executors.infer import Infer
from mlcomp.worker.reports.classification import ClassificationReportBuilder


@Executor.register
class InferMnist(Infer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.test:
            self.x = MnistDataset(file='data/test.csv')
        else:
            self.x = MnistDataset(
                file='data/train.csv', fold_csv='data/fold.csv', is_test=True
            )

    def submit(self, res):
        argmax = res.argmax(axis=1)
        pd.DataFrame(
            {
                'ImageId': np.arange(1,
                                     len(argmax) + 1),
                'Label': argmax
            }
        ).to_csv(self.out_file, index=False)

    def plot(self, res):
        imgs = [(row['features'][0] * 255).astype(np.uint8) for row in self.x]
        builder = ClassificationReportBuilder(
            session=self.session,
            task=self.task.id,
            layout=self.layout,
            preds=res,
            targets=None if self.test else self.x.y,
            imgs=imgs
        )
        builder.build()