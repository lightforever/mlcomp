from albumentations.augmentations.transforms import ImageOnlyTransform
import numpy as np


class ChannelTranspose(ImageOnlyTransform):
    def __init__(self, axes=(2, 0, 1)):
        super(ChannelTranspose, self).__init__(always_apply=True)
        self.axes = axes

    def apply(self, img, **params):
        return np.transpose(img, self.axes)
