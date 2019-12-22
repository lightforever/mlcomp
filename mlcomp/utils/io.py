import os
from io import BytesIO
from zipfile import ZipFile

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


def yaml_dump(data, file: str = None):
    res = yaml.dump(data, default_flow_style=False, sort_keys=False)
    if file:
        open(file, 'w').write(res)
    return res


def zip_folder(folder: str, dst: str = None):
    if dst is None:
        dst = BytesIO()

    with ZipFile(dst, 'w') as zip_obj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zip_obj.write(filePath, os.path.relpath(filePath, folder))
    return dst


__all__ = ['read_lines', 'from_module_path', 'yaml_load', 'yaml_dump',
           'zip_folder']
