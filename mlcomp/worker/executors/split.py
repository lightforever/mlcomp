from os.path import join

import pandas as pd

from mlcomp.utils.config import Config
from mlcomp.contrib.split import stratified_k_fold
from mlcomp.worker.executors import Executor


@Executor.register
class Split(Executor):
    __syn__ = 'split'

    def __init__(
        self,
        variant: str,
        out: str,
        n_splits: int = 5,
        file: str = None,
        label: str = None
    ):
        self.variant = variant
        self.file = file
        self.n_splits = n_splits
        self.out = out
        self.label = label

    def work(self):
        if self.variant == 'frame':
            fold = stratified_k_fold(
                file=self.file, n_splits=self.n_splits, label=self.label
            )
            df = pd.DataFrame({'fold': fold})
            df.to_csv(self.out, index=False)

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        file = join(config.data_folder, executor.get('file'))
        return cls(
            variant=executor['variant'],
            out=join(config.data_folder, 'fold.csv'),
            file=file
        )


__all__ = ['Split']
