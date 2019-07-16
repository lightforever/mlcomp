from os.path import basename, splitext

import pandas as pd

from sklearn.model_selection import GroupKFold


def file_group_kfold(n_splits: int,
                     output: str,
                     get_group=None,
                     sort=False,
                     must_equal = (),
                     **files):
    assert len(files) > 0, 'at lease 1 type of files is required'
    fold = GroupKFold(n_splits)
    keys = sorted(list(files))

    def get_name(file):
        return splitext(basename(file))[0]

    if sort:
        for k, v in files.items():
            files[k] = sorted(files[k], key=get_name)

    file_first = sorted(files[keys[0]])

    assert len(file_first) > n_splits, \
        f'at least {n_splits} files is required. Provided: {len(file_first)}'

    for k, v in files.items():
        assert len(files[k]) == len(file_first), \
            f'count of files in key = {k} is not the same as in {keys[0]}'

    for k, v in files.items():
        if k not in must_equal:
            continue
        for i in range(len(file_first)):
            names_equal = get_name(v[i]) == get_name(file_first[i])
            assert names_equal, \
                f'file name in {k} does not equal to {keys[0]}, ' \
                    f'file name = {basename(v[i])}'

    df = pd.DataFrame(files)[keys]
    df['fold'] = 0

    groups = [i if not get_group else get_group(file) for i, file in
              enumerate(file_first)]

    for i, (train_index, test_index) in enumerate(
            fold.split(groups, groups=groups)):
        df.loc[test_index, 'fold'] = i

    df = df.sample(frac=1)
    df.to_csv(output, index=False)
