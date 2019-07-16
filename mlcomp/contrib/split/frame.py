import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold


def split_frame(file: str, label: str = None, n_splits=5):
    df = pd.read_csv(file)
    index = np.arange(df.shape[0])
    res = np.zeros(index.shape)
    if label is not None:
        folds = StratifiedKFold(n_splits=n_splits,
                                shuffle=True).split(index, df[label])
    else:
        folds = KFold(n_splits=n_splits, shuffle=True).split(index)

    for i, (train, val) in enumerate(folds):
        res[val] = i
    return pd.DataFrame({'fold': res})
