import collections
import copy
from datetime import datetime
import re
from typing import List
import numpy as np

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')


def dict_func(objcts: List, func=np.mean):
    if len(objcts)==0:
        return {}
    first = objcts[0]

    res = dict()
    for k in first:
        k_objcts = [o[k] for o in objcts]
        if isinstance(first[k], dict):
            res[k] = dict_func(k_objcts, func)
        elif isinstance(first[k], list):
            res[k] = [dict_func([o[k][i] for o in objcts], func) for i in range(len(first[k]))]
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

if __name__=='__main__':
    print(dict_func([
        {'cpu': 10, 'gpu': [{'memory': 20, 'load': 30}, {'memory': 0, 'load': 0}]},
        {'cpu': 50, 'gpu': [{'memory': 100, 'load': 100}, {'memory': 0, 'load': 0}]},
    ]))