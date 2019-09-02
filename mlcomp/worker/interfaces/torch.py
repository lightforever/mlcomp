import numpy as np
from tqdm import tqdm

import torch
from torch.jit import load
from torch.utils.data import DataLoader

from mlcomp.worker.interfaces.base import Interface


@Interface.register
class Torch(Interface):
    def __init__(self, file: str, batch_size: int, batch_mode: bool = False,
                 use_logistic: bool = False, num_workers: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = load(file).cuda()

        self.batch_size = batch_size
        self.use_logistic = use_logistic
        self.batch_mode = batch_mode
        self.num_workers = num_workers

    def _batch_call(self, loader):
        for batch in tqdm(loader, total=len(loader)):
            features = batch['features'].cuda()
            logits = self.model(features)
            if self.use_logistic:
                # noinspection PyTypeChecker
                p = 1/(1+torch.exp(-logits))
            else:
                p = torch.softmax(logits, 1)

            p = p.detach().cpu().numpy()
            yield {'prob': p, 'count': p.shape[0], **batch}

    def _standard_call(self, loader):
        pred = []
        for batch in tqdm(loader, total=len(loader)):
            features = batch['features'].cuda()
            logits = self.model(features)
            if self.use_logistic:
                # noinspection PyTypeChecker
                p = 1 / (1 + torch.exp(-logits))
            else:
                p = torch.softmax(logits, 1)

            p = p.detach().cpu().numpy()
            pred.append(p)

        pred = np.vstack(pred)
        return {'prob': pred}

    def __call__(self, x: dict):
        loader = DataLoader(
            x['dataset'], batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        if self.batch_mode:
            return self._batch_call(loader)

        return self._standard_call(loader)
