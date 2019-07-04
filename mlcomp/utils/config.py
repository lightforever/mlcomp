from typing import List

import yaml
from collections import OrderedDict
import os
import json
import albumentations as A


class Config(OrderedDict):
    @property
    def data_folder(self):
        return os.path.join(self['info']['project'], 'data')

    @staticmethod
    def from_json(config: str):
        return Config(json.loads(config))

    @staticmethod
    def from_yaml(config: str):
        return load_ordered_yaml(text=config)


def load_ordered_yaml(file: str = None, text: str = None,
                      Loader=yaml.Loader,
                      object_pairs_hook=OrderedDict) -> Config:
    """
    Loads `yaml` config into OrderedDict
    Args:
        file: file with yaml
        text: file's yaml content
        Loader: base class for yaml Loader
        object_pairs_hook: type of mapping
    Returns:
        dict: configuration
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    stream = open(file, "r") if file else text
    return Config(yaml.load(stream, OrderedLoader))


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
