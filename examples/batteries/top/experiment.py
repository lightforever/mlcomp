from collections import OrderedDict
import torchvision
import numpy as np
from torchvision import transforms
from catalyst.dl.experiments import ConfigExperiment
from mlcomp.contrib.dataset.segment import ImagesWithMasksDataset
import albumentations as A
import cv2

class Experiment(ConfigExperiment):
    std = 0.25
    mean = 0.5

    @staticmethod
    def transforms_train():
        return A.Compose([
            A.Resize(height=307, width=950),
            A.RandomSizedCrop(min_max_height=(230, 307), height=288, width=928, w2h_ratio=950 / 307),
            A.HorizontalFlip(),
            A.JpegCompression(quality_lower=60, quality_upper=100),
            A.RandomGamma(),
            A.Normalize(mean=(Experiment.mean), std=(Experiment.std))
        ])

    @staticmethod
    def transforms_valid():
        return A.Compose(
            [
                A.Resize(288, 928),
                A.Normalize(mean=(Experiment.mean), std=(Experiment.std))
            ]
        )

    def denormilize(self, img: np.array):
        # ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        return (img[0] * Experiment.std + Experiment.mean) * 255


    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        params = self.stages_config[stage]['data_params']
        train = ImagesWithMasksDataset(params['image_folder'], params['mask_folder'], params['fold_csv'],
                                       params['fold_number'], is_test=False,
                                       gray_scale=True, num_classes=2, transforms=Experiment.transforms_train(),
                                       max_count=1,
                                       meta_cols=('cathode_count',)
                                       )

        valid = ImagesWithMasksDataset(params['image_folder'], params['mask_folder'], params['fold_csv'],
                                       params['fold_number'], is_test=True,
                                       gray_scale=True, num_classes=2, transforms=Experiment.transforms_valid(),
                                       max_count=1,
                                       meta_cols=('cathode_count',)
                                       )

        datasets['train'] = train
        datasets['valid'] = valid
        return datasets
