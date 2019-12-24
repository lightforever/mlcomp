import numpy as np
from albumentations import ImageOnlyTransform


class ChannelTranspose(ImageOnlyTransform):
    def get_transform_init_args_names(self):
        return ()

    def get_params_dependent_on_targets(self, params):
        pass

    def __init__(self, axes=(2, 0, 1)):
        super().__init__(always_apply=True)
        self.axes = axes

    def apply(self, img, **params):
        return np.transpose(img, self.axes)


class Ensure4d(ImageOnlyTransform):
    def get_transform_init_args_names(self):
        return ()

    def get_params_dependent_on_targets(self, params):
        pass

    def __init__(self):
        super().__init__(always_apply=True)

    def apply(self, img, **params):
        while len(img.shape) < 4:
            img = img[None]
        return img


__all__ = ['ChannelTranspose', 'Ensure4d']
