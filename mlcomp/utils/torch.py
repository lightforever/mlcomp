import numpy as np
from tqdm import tqdm

import torch
from torch.jit import load
from torch.utils.data import DataLoader, Dataset

from mlcomp.contrib.transform.tta import TtaWrap


def _infer_batch(model, loader: DataLoader, use_logistic):
    for batch in tqdm(loader, total=len(loader)):
        features = batch['features'].cuda()
        logits = model(features)
        if use_logistic:
            # noinspection PyTypeChecker
            p = 1 / (1 + torch.exp(-logits))
        else:
            p = torch.softmax(logits, 1)

        if isinstance(loader.dataset, TtaWrap):
            p = loader.dataset.inverse(p)
        p = p.detach().cpu().numpy()

        yield {'prob': p, 'count': p.shape[0], **batch}


def _infer(model, loader: DataLoader, use_logistic):
    pred = []
    for batch in tqdm(loader, total=len(loader)):
        features = batch['features'].cuda()
        logits = model(features)
        if use_logistic:
            # noinspection PyTypeChecker
            p = 1 / (1 + torch.exp(-logits))
        else:
            p = torch.softmax(logits, 1)

        if isinstance(loader.dataset, TtaWrap):
            p = loader.dataset.inverse(p)
        p = p.detach().cpu().numpy()

        pred.append(p)

    pred = np.vstack(pred)
    return pred


def infer(x: Dataset, file: str, batch_size: int = 1,
          batch_mode: bool = False,
          use_logistic: bool = True, num_workers: int = 1):
    loader = DataLoader(
        x, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    model = load(file).cuda()
    if batch_mode:
        return _infer_batch(model, loader, use_logistic=use_logistic)

    return _infer(model, loader, use_logistic=use_logistic)


__all__ = ['infer']
