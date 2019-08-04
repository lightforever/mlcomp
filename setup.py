#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

# Based on https://github.com/catalyst-team/catalyst/blob/master/setup.py

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'mlcomp'
DESCRIPTION = 'Machine learning pipelines. ' \
              'Especially, for competitions, like Kaggle'
URL = 'https://github.com/lightforever/mlcomp'
EMAIL = 'lightsanweb@gmail.com'
AUTHOR = 'Evgeny Semyonov'
REQUIRES_PYTHON = '>=3.6.0'

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements():
    with open(os.path.join(PROJECT_ROOT, 'requirements.txt'), 'r') as f:
        return f.read()


def load_readme():
    readme_path = os.path.join(PROJECT_ROOT, 'README.md')
    with io.open(readme_path, encoding='utf-8') as f:
        return '\n' + f.read()


def load_version():
    context = {}
    with open(os.path.join(PROJECT_ROOT, 'mlcomp', '__version__.py')) as f:
        exec(f.read(), context)
    return context['__version__']


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def get_package_data():
    res = ['utils/req_stdlib']
    res.extend(package_files('mlcomp/server/front/dist'))
    return res


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(PROJECT_ROOT, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(
                sys.executable
            )
        )

        self.status('Uploading the package to PyPI via Twine…')
        os.system('python -m twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(load_version()))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=load_version(),
    description=DESCRIPTION,
    long_description=load_readme(),
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests', )),
    install_requires=load_requirements(),
    include_package_data=True,
    package_data={'': get_package_data()},
    zip_safe=False,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
    entry_points={
        'console_scripts': [
            'mlcomp-server = mlcomp.server.__main__:main',
            'mlcomp-worker = mlcomp.worker.__main__:main',
            'mlcomp-contrib = mlcomp.contrib.__main__:main',
            'mlcomp = mlcomp.__main__:main',
        ],
    }
)
