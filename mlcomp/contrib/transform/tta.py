import albumentations as A
import numpy as np

from torch.utils.data import Dataset

from mlcomp.contrib.torch.tensors import flip


class TtaWrap(Dataset):
    def __init__(self, dataset: Dataset, tfms=()):
        self.dataset = dataset
        self.tfms = tfms

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def inverse(self, a: np.array):
        last_dim = len(a.shape) - 1
        for t in self.tfms:
            if isinstance(t, A.HorizontalFlip):
                a = flip(a, last_dim)
            elif isinstance(t, A.VerticalFlip):
                a = flip(a, last_dim - 1)
            elif isinstance(t, A.Transpose):
                axis = (0, 1, 3, 2) if len(a.shape) == 4 else (0, 2, 1)
                a = a.permute(*axis)

        return a


__all__ = ['TtaWrap']
