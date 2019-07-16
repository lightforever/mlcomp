from typing import List
import os
import ast
from glob import glob
import pathspec
import pkg_resources

from mlcomp.utils.logging import logger
from mlcomp.utils.io import read_lines

_mapping = {
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'migrate': 'sqlalchemy-migrate'
}


def find_imports(path: str,
                 files: List[str] = None,
                 exclude_patterns: List[str] = None,
                 encoding='utf-8'):
    res = []
    raw_imports = []
    files = files if files is not None \
        else glob(os.path.join(path, '**', '*.py'), recursive=True)

    exclude_patterns = exclude_patterns \
        if exclude_patterns is not None else []
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern,
                                        exclude_patterns)

    for file in files:
        if not file.endswith('.py'):
            continue
        file_rel = os.path.relpath(file, path)
        if spec.match_file(file_rel):
            continue

        with open(file, "r", encoding=encoding) as f:
            content = f.read()
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for subnode in node.names:
                            raw_imports.append((subnode.name, file_rel))
                    elif isinstance(node, ast.ImportFrom):
                        raw_imports.append((node.module, file_rel))
            except Exception as exc:
                logger.error("Failed on file: %s" % file_rel)
                raise exc

    for lib, file in raw_imports:
        name = lib.split('.')[0]
        try:
            if name in _mapping:
                name = _mapping[name]

            version = pkg_resources.get_distribution(name).version
            res.append((name, version))
        except Exception as exc:
            pass

    return res


def _read_requirements(file: str):
    res = []
    for l in read_lines(file):
        if l == '':
            continue
        name, rel, ver = None, None, None
        if '>=' in l:
            rel = '>='
        elif '==' in l:
            rel = '=='

        name = l.split(rel)[0].strip()
        if rel:
            ver = l.split(rel)[1].strip()

        res.append([name, rel, ver])

    return res


def _write_requirements(file: str, reqs: List):
    with open(file, 'w') as f:
        text = '\n'.join([f'{name}{rel}{ver}'
                          if rel else name for name, rel, ver in reqs])

        f.write(text)


def control_requirements(path: str,
                         files: List[str] = None,
                         exclude_patterns: List[str] = None):
    req_file = os.path.join(path, 'requirements.txt')
    if not os.path.exists(req_file):
        with open(req_file, 'w') as f:
            f.write('')

    req_ignore_file = os.path.join(path, 'requirements.ignore.txt')
    if not os.path.exists(req_ignore_file):
        with open(req_ignore_file, 'w') as f:
            f.write('')

    libs = find_imports(path, files=files, exclude_patterns=exclude_patterns)
    module_folder = os.path.dirname(__file__)
    stdlib_file = os.path.join(module_folder, 'req_stdlib')
    ignore_libs = set(read_lines(req_ignore_file) + read_lines(stdlib_file))

    reqs = _read_requirements(req_file)
    for lib, version in libs:
        if lib in ignore_libs:
            continue
        found = False
        for i in range(len(reqs)):
            if reqs[i][0] == lib:
                found = True
                reqs[i][1] = '>='
                reqs[i][2] = version
                break
        if not found:
            reqs.append([lib, '>=', version])

    _write_requirements(req_file, reqs)
    return reqs


if __name__ == '__main__':
    folder = '/home/light/projects/mlcomp/'
    # for l, v in find_imports(folder):
    #     print(f'{l}={v}')
    control_requirements(folder, exclude_patterns=['mlcomp/server/front'])
