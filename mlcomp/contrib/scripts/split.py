from sklearn.model_selection import GroupKFold
import pandas as pd


def file_group_kfold(n_splits: int, output: str, get_group=None, **files):
    fold = GroupKFold(n_splits)
    keys = sorted(list(files))
    for k, v in files.items():
        files[k] = sorted(files[k])

    df = pd.DataFrame({k: files[k] for k in keys})[keys]
    df['fold'] = 0
    groups = [i if not get_group else get_group(file) for i, file in enumerate(files[keys[0]])]
    for i, (train_index, test_index) in enumerate(fold.split(groups, groups=groups)):
        df.loc[test_index, 'fold'] = i

    df.to_csv(output, index=False)
