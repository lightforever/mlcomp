from collections import OrderedDict

from catalyst.dl import ConfigExperiment

from dataset import MnistDataset


class Experiment(ConfigExperiment):
    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train = MnistDataset('data/train.csv',
                             fold_csv='data/fold.csv')

        valid = MnistDataset('data/train.csv',
                             fold_csv='data/fold.csv',
                             is_test=True
                             )

        datasets['train'] = train
        datasets['valid'] = valid

        return datasets
