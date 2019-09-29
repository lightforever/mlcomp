from os.path import join

import cv2
import numpy as np

from mlcomp.contrib.dataset.classify import ImageDataset


class ImageWithMaskDataset(ImageDataset):
    def __init__(
            self, *, mask_folder: str, crop_positive=None, encode=True,
            include_binary=False, **kwargs
    ):
        assert mask_folder, 'Mask folder is required'
        self.mask_folder = mask_folder
        self.crop_positive = crop_positive
        self.encode = encode
        self.include_binary = include_binary
        if not encode and kwargs.get('num_classes', 1) > 1:
            kwargs['num_classes'] += 1

        super().__init__(**kwargs)

    def preprocess_row(self, row: dict):
        row['image'] = join(self.img_folder, row['image'])
        row['mask'] = join(self.mask_folder, row['mask'])

    def _get_item_before_transform(self, row: dict, item: dict):
        if 'mask' in row and self.mask_folder:
            item['mask'] = self.read_image_file(row['mask'], True)
            self._process_crop_positive(item)

    def _get_item_after_transform(
            self, row: dict, transformed: dict, res: dict
    ):
        if 'mask' in transformed:
            mask = transformed['mask'].astype(np.int64)
            if len(mask.shape) == 2:
                mask_encoded = np.zeros(
                    (self.num_classes, *mask.shape), dtype=np.float32
                )
                if self.num_classes == 1:
                    mask = (mask >= 1).astype(np.uint8)

                if self.encode:
                    for i in range(1, self.num_classes + 1):
                        mask_encoded[i - 1] = mask == i

                    mask = mask_encoded
                else:
                    for i in range(1, self.num_classes + 1):
                        mask_encoded[i - 1] = mask == i

                    res['targets_encoded'] = mask_encoded

            res['targets'] = mask.astype(np.float32)

            if self.include_binary:
                for i, c in enumerate(mask):
                    res[f'empty_{i}'] = int(c.sum() == 0)
                res['empty_all'] = int(mask.sum() == 0)

    def _process_crop_positive(self, item: dict):
        if not self.crop_positive:
            return
        mask = item['mask']
        size = mask.shape

        # import matplotlib.pyplot as plt
        # plt.imshow(item['image'])
        # plt.show()
        # plt.imshow(item['mask'] * 53)
        # plt.show()
        crop_pos_y = self.crop_positive[0]
        if type(crop_pos_y) == tuple:
            crop_pos_y = np.random.randint(crop_pos_y[0], crop_pos_y[1])

        crop_pos_x = self.crop_positive[1]
        if type(crop_pos_x) == tuple:
            crop_pos_x = np.random.randint(crop_pos_x[0], crop_pos_x[1])

        p = self.crop_positive[2]

        if mask.sum() == 0 or np.random.uniform(0, 1) < p:
            min_y = 0
            max_y = size[0] - crop_pos_y
            min_x = 0
            max_x = size[1] - crop_pos_x
        else:
            mask_binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = contours[np.random.randint(0, len(contours))]
            rect = cv2.boundingRect(contour)
            min_y = rect[1] - crop_pos_y + int(rect[3] // 2)
            max_y = rect[1] + int(rect[3] // 2)

            min_x = rect[0] - crop_pos_x + int(rect[2] // 2)
            max_x = rect[0] + int(rect[2] // 2)

        min_x = max(0, min_x)
        min_y = max(0, min_y)

        max_x = min(max_x, size[1] - crop_pos_x)
        max_y = min(max_y, size[0] - crop_pos_y)

        x = np.random.randint(min_x, max_x + 1)
        y = np.random.randint(min_y, max_y + 1)
        item['image'] = item['image'][y:y + crop_pos_y, x:x + crop_pos_x]

        item['mask'] = item['mask'][y:y + crop_pos_y, x:x + crop_pos_x]

        # plt.imshow(item['image'])
        # plt.show()
        # plt.imshow(item['mask'] * 53)
        # plt.show()
