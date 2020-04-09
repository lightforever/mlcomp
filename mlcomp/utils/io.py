import os
from os.path import join

from typing import List

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import yaml

yaml.warnings({'YAMLLoadWarning': False})


def read_pandas(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        raise Exception('Unknown file type')
    return df


def read_lines(file: str):
    return [l.strip() for l in open(file)]


def from_module_path(file: str, path: str):
    return os.path.join(os.path.dirname(file), path)


def yaml_load(text: str = None, file: str = None):
    stream = text or ''
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


def zip_folder(
        folder: str = None,
        dst: str = None,
        folders: List[str] = (),
        files: List[str] = (),
        root: bool = None
):
    if root is None and len(folders) > 0:
        root = True
    if dst is None:
        dst = BytesIO()
    if folder:
        folders = (folder,)

    with ZipFile(dst, 'w') as zip_obj:
        # Iterate over all the files in directory
        for folder in folders:
            for folderName, subfolders, filenames in os.walk(folder):
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = join(folderName, filename)
                    # Add file to zip
                    rel_path = os.path.relpath(filePath, folder)
                    if root:
                        rel_path = join(os.path.basename(folder), rel_path)
                    zip_obj.write(filePath, rel_path)

        for file in files:
            zip_obj.write(file, file)
    return dst


__all__ = ['read_lines', 'from_module_path', 'yaml_load', 'yaml_dump',
           'zip_folder']
