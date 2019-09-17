from collections import OrderedDict

import albumentations as A

from catalyst.dl import ConfigExperiment

from dataset import MnistDataset


class Experiment(ConfigExperiment):
    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train = MnistDataset(
            'data/train.csv',
            fold_csv='data/fold.csv',
            transforms=Experiment.get_test_transforms()
        )

        valid = MnistDataset(
            'data/train.csv',
            fold_csv='data/fold.csv',
            is_test=True,
            transforms=Experiment.get_test_transforms()
        )

        datasets['train'] = train
        datasets['valid'] = valid

        return datasets

    @staticmethod
    def get_test_transforms():
        return A.Compose([A.Normalize(mean=(0.485, ), std=(0.229, ))])
