import collections
import copy
from datetime import datetime
import re
from typing import List
import os
import pwd
import random

import dateutil
import numpy as np
import signal
import psutil
from subprocess import check_output

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.

    Args:
        seed: random seed
    """

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)


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


def get_pid(name):
    res = []
    for line in check_output(["pgrep", '-la', name]).decode().split('\n'):
        parts = line.split()
        res.append((int(parts[0]), parts[1:]))
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


def adapt_db_types(d):
    dic = d.__dict__ if not isinstance(d, dict) else d
    for k in dic:
        if type(dic[k]) in [np.int64]:
            dic[k] = int(dic[k])
        elif type(dic[k]) in [np.float64]:
            dic[k] = float(dic[k])


def dict_flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(dict_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_unflatten(d: dict, sep='/'):
    res = dict()
    for key, value in d.items():
        parts = key.split(sep)
        c = res
        for part in parts[:-1]:
            if part not in c:
                c[part] = dict()
            c = c[part]
        c[parts[-1]] = value

    return res


def memory():
    return map(int, os.popen('free -t -m').readlines()[1].split()[1:4])


def disk(folder: str):
    filesystem, total, used, available, use, mounded_point \
        = os.popen(f'df {folder}').readlines()[1].split()
    total = int(int(total) / 10 ** 6)
    available = int(int(available) / 10 ** 6)
    use = int(use[:-1])
    return total, use, available


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def parse_time(time):
    if not time:
        return None
    if isinstance(time, str):
        return dateutil.parser.parse(time)
    return time


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)

    parent.send_signal(sig)


if __name__ == '__main__':
    print(dict_unflatten({'a/b': 10, 'a/e': 20, 'a/c/d': 40}))
