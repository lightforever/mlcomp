from typing import List
import os
import json

import albumentations as A

from mlcomp.utils.misc import yaml_load
from mlcomp.utils.settings import DATA_FOLDER


class Config(dict):
    @property
    def data_folder(self):
        return os.path.join(DATA_FOLDER, self['info']['project'])

    @staticmethod
    def from_json(config: str):
        return Config(json.loads(config))

    @staticmethod
    def from_yaml(config: str):
        return yaml_load(config)


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
