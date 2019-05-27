import os


def read_lines(file: str):
    return [l.strip() for l in open(file)]


def from_module_path(file: str, path: str):
    return os.path.join(os.path.dirname(file), path)
