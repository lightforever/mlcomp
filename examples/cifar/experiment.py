from collections import OrderedDict

import numpy as np

import torchvision
from torchvision import transforms

from catalyst.dl import ConfigExperiment


class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def denormilize(self, img: np.array):
        # ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

        res = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        for i in range(res.shape[2]):
            res[:, :, i] = (img[i] * 0.5 + 0.5) * 255
        return res

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode='train')
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode='valid')
        )

        datasets['train'] = trainset
        datasets['valid'] = testset

        return datasets
