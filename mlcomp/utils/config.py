from collections import defaultdict
from typing import List
import os
import json

import albumentations as A

from mlcomp import DATA_FOLDER
from mlcomp.utils.io import yaml_load
from mlcomp.utils.misc import dict_flatten, dict_unflatten


class Config(dict):
    @property
    def data_folder(self):
        return os.path.join(DATA_FOLDER, self['info']['project'])

    @staticmethod
    def from_json(config: str):
        return Config(json.loads(config))

    @staticmethod
    def from_yaml(config: str):
        return Config(yaml_load(config))


def merge_dicts_smart(target: dict, source: dict, sep='/'):
    target_flatten = dict_flatten(target)
    mapping = defaultdict(list)
    hooks = dict()

    for k, v in target_flatten.items():
        parts = k.split(sep)
        for i in range(len(parts) - 1, -1, -1):
            key = sep.join(parts[i:])
            mapping[key].append(k)

            if i < len(parts) - 1:
                hooks[sep.join(parts[i: -1])] = sep.join(parts[:i + 1])

    for k, v in list(source.items()):
        if isinstance(v, dict):
            source.update({k + sep + kk: v for kk in dict_flatten(v)})

    for k, v in source.items():
        if len(mapping[k]) == 0:
            parts = k.split(sep)
            hook = None
            for i in range(len(parts) - 1, -1, -1):
                h = sep.join(parts[:i])
                if h in hooks:
                    hook = hooks[h] + sep + sep.join(parts[i:])
                    break

            if not hook:
                hook = k

            mapping[k] = [hook]
        assert len(mapping[k]) == 1, f'ambiguous mapping for {k}'
        key = mapping[k][0]
        target_flatten[key] = v

    return dict_unflatten(target_flatten)


def parse_albu(configs: List[dict]):
    res = []
    for config in configs:
        assert 'name' in config, f'name is required in {config}'
        config = config.copy()
        name = config.pop('name')
        if name == 'Compose':
            items = config.pop('items')
            aug = A.Compose(parse_albu(items), **config)
        else:
            aug = getattr(A, name)(**config)
        res.append(aug)
    return res


__all__ = ['Config', 'merge_dicts_smart', 'parse_albu']
