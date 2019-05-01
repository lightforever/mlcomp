import collections
import copy
from datetime import datetime
import re

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')

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