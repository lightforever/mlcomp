from os.path import join
from numbers import Number
from collections import defaultdict

import tifffile
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
            self,
            *,
            img_folder: str,
            fold_csv: str,
            fold_number: int,
            is_test: bool = False,
            gray_scale: bool = False,
            num_classes=2,
            max_count=None,
            meta_cols=(),
            transforms=None):
        self.img_folder = img_folder

        df = pd.read_csv(fold_csv)
        if is_test:
            self.data = df[df['fold'] == fold_number].to_dict(orient='row')
        else:
            self.data = df[df['fold'] != fold_number].to_dict(orient='row')

        self.data = self.data
        if max_count is not None:
            self.apply_max_count(max_count)

        for row in self.data:
            self.preprocess_row(row)

        self.transforms = transforms
        self.gray_scale = gray_scale
        self.num_classes = num_classes
        self.meta_cols = meta_cols

    def apply_max_count(self, max_count):
        if isinstance(max_count, Number):
            self.data = self.data[:max_count]
        else:
            data = defaultdict(list)
            for row in self.data:
                data[row['label']].append(row)
            min_index = np.argmin(max_count)
            min_count = len(data[min_index])
            for k, v in data.items():
                count = int(min_count*(max_count[k]/max_count[min_index]))
                data[k] = data[k][:count]

            self.data = [v for i in range(len(data)) for v in data[i]]

    def preprocess_row(self, row: dict):
        row['image'] = join(self.img_folder, str(row['label']), row['image'])

    def __len__(self):
        return len(self.data)

    def _get_item_before_transform(self, row: dict, item: dict):
        pass

    def _get_item_after_transform(self, row: dict,
                                  transformed: dict,
                                  res: dict):
        res['targets'] = row['label']

    def __getitem__(self, index):
        row = self.data[index]
        image = self.read_image_file(row['image'], self.gray_scale)
        item = {'image': image}

        self._get_item_before_transform(row, item)

        if self.transforms:
            item = self.transforms(**item)
        if self.gray_scale:
            item['image'] = np.expand_dims(item['image'], axis=0)
        res = {
            'features': item['image'].astype(np.float32),
            'meta': {c: row[c] for c in self.meta_cols}
        }
        self._get_item_after_transform(row, item, res)

        return res

    @staticmethod
    def read_image_file(path: str, gray_scale=False):
        if path.endswith('.tiff') and not gray_scale:
            return tifffile.imread(path)
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            if gray_scale:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                assert img is not None, \
                    f'Image at path {path} does not exist'
                return img.astype(np.uint8)
            else:
                img = cv2.imread(path)
                assert img is not None, \
                    f'Image at path {path} does not exist'
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
