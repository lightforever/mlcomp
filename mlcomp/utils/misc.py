import collections
import copy
from datetime import datetime
import re
from typing import List
import os
import yaml

yaml.warnings({'YAMLLoadWarning': False})

import numpy as np

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def yaml_load(text: str = None, file: str = None):
    stream = text
    if file is not None:
        stream = open(file).read()
    return yaml.load(stream, Loader=yaml.FullLoader) or {}


def yaml_dump(data):
    return yaml.dump(data,
                     default_flow_style=False,
                     sort_keys=False)


def dict_func(objcts: List, func=np.mean):
    if len(objcts) == 0:
        return {}
    first = objcts[0]

    res = dict()
    for k in first:
        k_objcts = [o[k] for o in objcts]
        if isinstance(first[k], dict):
            res[k] = dict_func(k_objcts, func)
        elif isinstance(first[k], list):
            res[k] = [dict_func([o[k][i] for o in objcts], func) for i in
                      range(len(first[k]))]
        else:
            res[k] = func(k_objcts)
    return res


def now():
    return datetime.utcnow()


def merge_dicts(*dicts: dict) -> dict:
    """
    Recursive dict merge.
    Instead of updating only top-level keys,
    ``merge_dicts`` recurses down into dicts nested
    to an arbitrary depth, updating keys.
    Args:
        *dicts: several dictionaries to merge
    Returns:
        dict: deep-merged dictionary
    """
    assert len(dicts) > 1

    dict_ = copy.deepcopy(dicts[0])

    for merge_dict in dicts[1:]:
        merge_dict = merge_dict or {}
        for k, v in merge_dict.items():
            if (
                    k in dict_ and isinstance(dict_[k], dict)
                    and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def to_snake(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()


def log_name(level: int):
    if level == 10:
        return 'DEBUG'
    if level == 20:
        return 'INFO'
    if level == 30:
        return 'WARNING'
    if level == 40:
        return 'ERROR'

    raise Exception('Unknown log level')


def duration_format(delta: float):
    """
    Duration format
    :param delta: seconds
    :return: string representation: 1 days 1 hour 1 min
    """
    if delta < 0:
        delta = f'{int(delta * 1000)} ms'
    elif delta < 60:
        delta = f'{int(delta)} sec'
    elif delta < 3600:
        delta = f'{int(delta / 60)} min {int(delta % 60)} sec'
    elif delta < 3600 * 24:
        hour = int(delta / 3600)
        delta = f'{hour} {"hours" if hour > 1 else "hour"} ' \
            f'{int((delta % 3600) / 60)} min'
    else:
        day = int(delta / (3600 * 24))
        hour = int((delta % (3600 * 24)) / 3600)
        delta = f'{day} {"days" if day > 1 else "day"} {hour}' \
            f' {"hours" if hour > 1 else "hour"} ' \
            f'{int((delta % 3600) / 60)} min'
    return delta


def adapt_db_types(d: dict):
    for k in d:
        if type(d[k]) in [np.int, np.int64]:
            d[k] = int(d[k])
        elif type(d[k]) in [np.float, np.float64]:
            d[k] = float(d[k])


def memory():
    return map(int, os.popen('free -t -m').readlines()[1].split()[1:4])


def disk(folder: str):
    filesystem, total, used, available, use, mounded_point \
        = os.popen(f'df {folder}').readlines()[1].split()
    total = int(int(total) / 10 ** 6)
    available = int(int(available) / 10 ** 6)
    use = int(use[:-1])
    return total, use, available


if __name__ == '__main__':
    print(dict_func([
        {'cpu': 10,
         'gpu': [{'memory': 20, 'load': 30}, {'memory': 0, 'load': 0}]},
        {'cpu': 50,
         'gpu': [{'memory': 100, 'load': 100}, {'memory': 0, 'load': 0}]},
    ]))
