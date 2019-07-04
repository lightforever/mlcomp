import os
import numpy as np
from mlcomp.contrib.dataset.classify import ImageDataset
from os.path import join


class ImageWithMaskDataset(ImageDataset):
    def __init__(self, *, mask_folder: str, **kwargs):
        assert mask_folder, 'Mask folder is required'
        self.mask_folder = mask_folder
        super().__init__(**kwargs)

    def preprocess_row(self, row: dict):
        row['image'] = join(self.img_folder, row['image'])
        row['mask'] = join(self.mask_folder, row['mask'])

    def _get_item_before_transform(self, row: dict, item: dict):
        if 'mask' in row and self.mask_folder:
            item['mask'] = self.read_image_file(row['mask'], True)

    def _get_item_after_transform(self, row: dict,
                                  transformed: dict, res: dict):
        if 'mask' in transformed:
            mask = transformed['mask']
            if len(mask.shape) == 2:
                mask_encoded = np.zeros((self.num_classes, *mask.shape),
                                        dtype=mask.dtype)
                for i in range(self.num_classes):
                    mask_encoded[i] = mask == i

                mask = mask_encoded
            res['targets'] = mask.astype(np.float32)
