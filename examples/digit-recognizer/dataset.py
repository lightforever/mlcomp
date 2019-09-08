import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(
        self,
        file: str,
        fold_csv: str = None,
        fold_number: int = 0,
        is_test: bool = False,
        max_count: int = None,
        transforms=None
    ):
        df = pd.read_csv(file)
        if fold_csv is not None:
            fold = pd.read_csv(fold_csv)
            if is_test:
                df = df[fold['fold'] == fold_number]
            else:
                df = df[fold['fold'] != fold_number]

        if max_count:
            df = df[:max_count]

        if 'label' in df.columns:
            self.y = df['label'].values.astype(np.int)
            df = df.drop(columns='label', axis=0)
        else:
            self.y = None

        self.x = df.values.reshape((-1, 1, 28, 28)).astype(np.float32)
        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        if self.transforms:
            x = self.transforms(image=x)['image']
        res = {'features': x}

        if self.y is not None:
            res['targets'] = self.y[index]
        return res
