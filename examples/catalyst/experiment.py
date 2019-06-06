from collections import OrderedDict
import torchvision
import numpy as np
from torchvision import transforms
from catalyst.dl.experiments import ConfigExperiment
import time

class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def denormilize(self, img: np.array):
        #``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        res = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
        for i in range(res.shape[2]):
            res[:,:,i] = (img[i]*0.5+0.5)*255
        return res

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode="train")
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode="valid")
        )

        # raise Exception()

        count = 5000
        trainset.train_data = trainset.train_data[:count]
        trainset.train_labels = np.clip(trainset.train_labels[:count], 0, 1)

        testset.train_data = trainset.train_data[:count]
        testset.train_labels = np.clip(trainset.train_labels[:count], 0, 1)

        testset.test_data = testset.test_data[:count]
        testset.test_labels = np.clip(testset.test_labels[:count], 0, 1)

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets