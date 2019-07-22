import os
import yaml

yaml.warnings({'YAMLLoadWarning': False})


def read_lines(file: str):
    return [l.strip() for l in open(file)]


def from_module_path(file: str, path: str):
    return os.path.join(os.path.dirname(file), path)


def yaml_load(text: str = None, file: str = None):
    stream = text
    if file is not None:
        stream = open(file).read()
    res = yaml.load(stream, Loader=yaml.FullLoader)
    if res is None:
        return {}
    return res


def yaml_dump(data):
    return yaml.dump(data,
                     default_flow_style=False,
                     sort_keys=False)


__all__ = ['read_lines', 'from_module_path', 'yaml_load', 'yaml_dump']