import re
import os
from os.path import join
from uuid import uuid4

import click
import pandas as pd

from mlcomp.contrib.scripts.split import file_group_kfold

current_folder = os.getcwd()


@click.group()
def main():
    pass


@main.command()
@click.argument('path')
@click.option('--n_splits', type=int, default=5)
def split_pandas(path: str, n_splits: int):
    output = join(current_folder, 'fold.csv')
    df = pd.read_csv(path)
    folds = file_group_kfold(n_splits,
                             image=df[df.columns[0]]
                             )
    df['fold'] = folds['fold']
    df.to_csv(output)


@main.command()
@click.argument('img_path')
@click.option('--n_splits', type=int, default=5)
@click.option('--group-regex')
def split_classify(img_path: str,
                   n_splits: int,
                   group_regex: str = None):
    output = join(current_folder, 'fold.csv')
    get_group = None
    if group_regex:
        pattern = re.compile(group_regex)

        def get_group(x):
            match = pattern.match(x)
            if not match:
                return str(uuid4())
            return match.group(1)

    images_labels = [(img, sub_folder)
                     for sub_folder in os.listdir(img_path)
                     for img in os.listdir(join(img_path, sub_folder))]

    file_group_kfold(n_splits, output, get_group=get_group,
                     image=[join(label, img) for img, label in images_labels],
                     label=[label for img, label in images_labels]
                     )


@main.command()
@click.argument('img_path')
@click.argument('mask_path')
@click.option('--n_splits', type=int, default=5)
@click.option('--group-regex')
def split_segment(img_path: str,
                  mask_path: str,
                  n_splits: int,
                  group_regex: str = None):
    output = join(current_folder, 'fold.csv')
    get_group = None
    if group_regex:
        pattern = re.compile(group_regex)

        def get_group(x):
            match = pattern.match(x)
            if not match:
                return str(uuid4())
            return match.group(1)

    file_group_kfold(n_splits, output, get_group=get_group,
                     image=os.listdir(img_path),
                     mask=os.listdir(mask_path),
                     sort=True,
                     must_equal=['image', 'mask']
                     )


@main.command()
@click.argument('img_path')
def split_test_img(img_path: str):
    output = join(current_folder, 'fold_test.csv')
    df = pd.DataFrame({
        'image': sorted(list(os.listdir(img_path))),
        'fold': 0
    })
    df.to_csv(output, index=False)


if __name__ == '__main__':
    main()
