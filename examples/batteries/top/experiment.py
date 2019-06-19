from collections import OrderedDict
import numpy as np
from catalyst.dl.experiments import ConfigExperiment
from mlcomp.contrib.dataset.segment import ImageWithMaskDataset
import albumentations as A
from mlcomp.contrib.transform.albumentations import ChannelTranspose

class Experiment(ConfigExperiment):
    std = 0.25
    mean = 0.5

    @staticmethod
    def transforms_train():
        return A.Compose([
            A.Resize(height=307, width=950),
            A.RandomSizedCrop(min_max_height=(230, 307), height=256, width=896, w2h_ratio=950 / 307),
            A.HorizontalFlip(),
            # A.JpegCompression(quality_lower=60, quality_upper=100),
            # A.RandomGamma(),
            A.Normalize(mean=(Experiment.mean, Experiment.mean, Experiment.mean), std=(Experiment.std, Experiment.std, Experiment.std)),
            ChannelTranspose()
        ])

    @staticmethod
    def transforms_valid():
        return A.Compose(
            [
                A.Resize(256, 896),
                A.Normalize(mean=(Experiment.mean, Experiment.mean, Experiment.mean), std=(Experiment.std, Experiment.std, Experiment.std)),
                ChannelTranspose()
            ]
        )

    @staticmethod
    def transforms_classify_valid():
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(),
                ChannelTranspose()
            ]
        )

    def denormilize(self, img: np.array):
        # ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        return (img[0] * Experiment.std + Experiment.mean) * 255

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        params = self.stages_config[stage]['data_params']
        common = {
            'img_folder': params['image_folder'],
            'mask_folder': params['mask_folder'],
            'fold_csv': params['fold_csv'],
            'fold_number': params['fold_number'],
            #'gray_scale': True,
            'num_classes': 3,
            #'max_count': 100,
            'meta_cols': ('cathode_count',)
        }

        train = ImageWithMaskDataset(**common, is_test=False, transforms=Experiment.transforms_train())
        valid = ImageWithMaskDataset(**common, is_test=True, transforms=Experiment.transforms_valid())

        datasets['train'] = train
        datasets['valid'] = valid
        return datasets
