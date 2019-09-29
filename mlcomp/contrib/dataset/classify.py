import ast
from os.path import join
from numbers import Number
from collections import defaultdict
import os
from typing import Callable, Dict

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
            fold_csv: str = None,
            fold_number: int = None,
            is_test: bool = False,
            gray_scale: bool = False,
            num_classes=2,
            max_count=None,
            meta_cols=(),
            transforms=None,
            postprocess_func: Callable[[Dict], Dict] = None,
            include_image_orig=False
    ):
        self.img_folder = img_folder

        if fold_csv:
            df = pd.read_csv(fold_csv)
            if fold_number is not None:
                if is_test:
                    self.data = df[df['fold'] == fold_number]
                else:
                    self.data = df[df['fold'] != fold_number]
            else:
                self.data = df
        else:
            self.data = pd.DataFrame(
                {'image': os.listdir(img_folder)}).sort_values(by='image')

        self.data = self.data.to_dict(orient='row')
        if max_count is not None:
            self.apply_max_count(max_count)

        for row in self.data:
            self.preprocess_row(row)

        self.transforms = transforms
        self.gray_scale = gray_scale
        self.num_classes = num_classes
        self.meta_cols = meta_cols
        self.postprocess_func = postprocess_func
        self.include_image_orig = include_image_orig

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
                count = int(min_count * (max_count[k] / max_count[min_index]))
                data[k] = data[k][:count]

            self.data = [v for i in range(len(data)) for v in data[i]]

    def preprocess_row(self, row: dict):
        row['image'] = join(self.img_folder, row['image'])

    def __len__(self):
        return len(self.data)

    def _get_item_before_transform(self, row: dict, item: dict):
        pass

    def _get_item_after_transform(self, row: dict,
                                  transformed: dict,
                                  res: dict):
        if 'label' in row:
            res['targets'] = ast.literal_eval(str(row['label']))
            if isinstance(res['targets'], list):
                res['targets'] = np.array(res['targets'], dtype=np.float32)

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
            'image_file': row['image']
        }
        if self.include_image_orig:
            res['image'] = image

        for c in self.meta_cols:
            res[c] = row[c]

        self._get_item_after_transform(row, item, res)
        if self.postprocess_func:
            res = self.postprocess_func(res)
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
