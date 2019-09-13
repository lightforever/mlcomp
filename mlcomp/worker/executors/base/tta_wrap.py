import albumentations as A
import numpy as np

from torch.utils.data import Dataset


class TtaWrap(Dataset):
    def __init__(self, dataset: Dataset, tfms=()):
        self.dataset = dataset
        self.tfms = tfms

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def inverse(self, a: np.array):
        for t in self.tfms:
            if isinstance(t, A.HorizontalFlip):
                a = a[:, :, :, ::-1]
            elif isinstance(t, A.VerticalFlip):
                a = a[:, :, ::-1]

        return a


__all__ = ['TtaWrap']
