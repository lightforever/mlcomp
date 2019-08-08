from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold


def stratified_group_k_fold(
    label: str,
    group_column: str,
    df: pd.DataFrame = None,
    file: str = None,
    n_splits=5
):
    if file is not None:
        df = pd.read_csv(file)

    labels = defaultdict(set)
    for g, l in zip(df[group_column], df[label]):
        labels[g].add(l)

    group_labels = dict()
    groups = []
    Y = []
    for k, v in labels.items():
        group_labels[k] = np.random.choice(list(v))
        Y.append(group_labels[k])
        groups.append(k)

    index = np.arange(len(group_labels))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True).split(index, Y)

    group_folds = dict()
    for i, (train, val) in enumerate(folds):
        for j in val:
            group_folds[groups[j]] = i

    res = np.zeros(len(df))
    for i, g in enumerate(df[group_column]):
        res[i] = group_folds[g]

    return res.astype(np.int)


def stratified_k_fold(
    label: str, df: pd.DataFrame = None, file: str = None, n_splits=5
):
    if file is not None:
        df = pd.read_csv(file)

    index = np.arange(df.shape[0])
    res = np.zeros(index.shape)
    folds = StratifiedKFold(n_splits=n_splits,
                            shuffle=True).split(index, df[label])

    for i, (train, val) in enumerate(folds):
        res[val] = i
    return res.astype(np.int)


__all__ = ['stratified_group_k_fold', 'stratified_k_fold']
