import numpy as np
from tqdm import tqdm

import torch
from torch.jit import load
from torch.utils.data import DataLoader, Dataset

from mlcomp.contrib.transform.tta import TtaWrap


def apply_activation(x, activation):
    if not activation:
        return x
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    if activation == 'softmax':
        return torch.softmax(x, 1)
    raise Exception(f'unknown activation = {activation}')


def _infer_batch(model, loader: DataLoader, activation=None):
    for batch in tqdm(loader, total=len(loader)):
        features = batch['features'].cuda()
        logits = model(features)
        p = apply_activation(logits, activation)

        if isinstance(loader.dataset, TtaWrap):
            p = loader.dataset.inverse(p)
        p = p.detach().cpu().numpy()

        yield {'prob': p, 'count': p.shape[0], **batch}


def _infer(model, loader: DataLoader, activation=None):
    pred = []
    for batch in tqdm(loader, total=len(loader)):
        features = batch['features'].cuda()
        logits = model(features)
        p = apply_activation(logits, activation)

        if isinstance(loader.dataset, TtaWrap):
            p = loader.dataset.inverse(p)
        p = p.detach().cpu().numpy()

        pred.append(p)

    pred = np.vstack(pred)
    return pred


def infer(
        x: Dataset,
        file: str,
        batch_size: int = 1,
        batch_mode: bool = False,
        activation=None,
        num_workers: int = 1,
):
    loader = DataLoader(
        x,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    model = load(file).cuda()
    if batch_mode:
        return _infer_batch(model, loader, activation=activation)

    return _infer(model, loader, activation=activation)


__all__ = ['infer', 'apply_activation']
