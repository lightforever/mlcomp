from itertools import product
from os.path import join
from typing import List
from glob import glob

from mlcomp.utils.io import yaml_load
from mlcomp.utils.misc import dict_flatten


def cell_name(cell: dict):
    c = dict_flatten(cell)
    parts = []
    for k, v in c.items():
        parts.append(f'{k}={v}')

    return ' '.join(parts)[-300:]


def grid_cells(grid: List):
    for i, row in enumerate(grid):
        row_type = type(row)

        if row_type == list:
            if len(row) == 0:
                raise Exception(f'Empty list at {i} position')
            if type(row[0]) != dict:
                raise Exception('List entries can be dicts only')
        elif row_type == dict:
            if len(row) != 1:
                raise Exception('Dict must contain only one element')
            key = list(row)[0]
            val_type = type(row[key])
            if val_type not in [list, str]:
                raise Exception('Dict value must be list or str')
            new_row = []
            if val_type == str:
                if '-' in row[key]:
                    start, end = map(int, row[key].split('-'))
                    for p in range(start, end + 1):
                        new_row.append({key: p})
                else:
                    if key == '_folder':
                        for file in glob(join(row[key], '*.yml')):
                            new_row.append(yaml_load(file))
            else:
                for v in row[key]:
                    if key == '_file':
                        new_row.append(yaml_load(v))
                    else:
                        new_row.append({key: v})

            grid[i] = new_row
        else:
            raise Exception(f'Unknown type of row = {row_type}')

    res = list(product(*grid))
    for i, r in enumerate(res):
        d = {}
        for dd in r:
            d.update(dd)
        res[i] = d
    return [[r, cell_name(r)] for r in res]


__all__ = ['grid_cells']
