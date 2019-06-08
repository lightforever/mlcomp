import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import tifffile
import pandas as pd


class ImagesWithMasksDataset(Dataset):
    def __init__(
            self,
            img_folder: str,
            mask_folder: str,
            fold_csv: str,
            fold_number: int,
            is_test: bool,
            gray_scale: bool = False,
            num_classes=1,
            max_count=None,
            meta_cols=(),
            transforms=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder

        df = pd.read_csv(fold_csv)
        if is_test:
            data = df[df['fold'] == fold_number].to_dict(orient='row')
        else:
            data = df[df['fold'] != fold_number].to_dict(orient='row')
        self.data = [{k: self.join_path(k, v) for k, v in row.items()} for row in data]
        if max_count:
            self.data = self.data[:max_count]
        self.transforms = transforms
        self.gray_scale = gray_scale
        self.num_classes = num_classes
        self.meta_cols = meta_cols

    def join_path(self, k: str, v):
        if k == 'image':
            return os.path.join(self.img_folder, v)
        if k == 'mask':
            return os.path.join(self.mask_folder, v)
        return v

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]
        image = self.read_image_file(d['image'], self.gray_scale)
        mask = self.read_image_file(d['mask'], self.gray_scale)
        res = {'image': image, 'mask': mask}
        if self.transforms:
            res = self.transforms(**res)
        if self.gray_scale:
            res['image'] = np.expand_dims(res['image'], axis=0)

        mask = res['mask']
        if len(mask.shape) == 2:
            mask_encoded = np.zeros((self.num_classes + 1, *mask.shape), dtype=mask.dtype)
            for i in range(self.num_classes + 1):
                mask_encoded[i] = mask == i

            mask = mask_encoded
        return {
            'features': res['image'].astype(np.float32),
            'targets': mask.astype(np.float32),
            'meta': {c: d[c] for c in self.meta_cols}
        }

    @staticmethod
    def read_image_file(path: str, gray_scale=False):
        if path.endswith('.tiff') and not gray_scale:
            return tifffile.imread(path)
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            if gray_scale:
                return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            else:
                return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
