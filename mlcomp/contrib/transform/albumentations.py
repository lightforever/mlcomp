import numpy as np

from albumentations.augmentations.transforms import ImageOnlyTransform


class ChannelTranspose(ImageOnlyTransform):
    def __init__(self, axes=(2, 0, 1)):
        super(ChannelTranspose, self).__init__(always_apply=True)
        self.axes = axes

    def apply(self, img, **params):
        return np.transpose(img, self.axes)
