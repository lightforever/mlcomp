import yaml
from collections import OrderedDict
import os


class Config(OrderedDict):
    @property
    def data_folder(self):
        return os.path.join(self['info']['folder'], 'data')


def load_ordered_yaml(file, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Loads `yaml` config into OrderedDict
    Args:
        file: file with yaml
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
    return Config(yaml.load(open(file, "r"), OrderedLoader))
