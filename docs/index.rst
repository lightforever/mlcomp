MLComp
======================================

.. image:: https://travis-ci.org/lightforever/mlcomp.svg?branch=master
    :target: https://travis-ci.org/lightforever/mlcomp
    :alt: Build Status

.. image:: https://img.shields.io/github/license/catalyst-team/mlcomp.svg
    :alt: License

.. image:: https://img.shields.io/pypi/v/mlcomp.svg
    :target: https://pypi.org/project/mlcomp/
    :alt: Pypi version

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fmlcomp%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v
    :target: https://catalyst-team.github.io/mlcomp/
    :alt: Docs


.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/MLcomp.png
    :target: https://github.com/catalyst-team/mlcomp

MLComp is a distributed DAG (Directed acyclic graph) framework for machine learning with UI.

The goal of MLComp is to provide tools for training, inference, creating complex pipelines
(especially for computer vision) in a rapid, well manageable, way.

MLComp is compatible with: Python 3.6+, Unix operation system.

**Features**

- Amazing UI
- `Catalyst <https://github.com/catalyst-team/catalyst>`_ support
- Distributed training
- Supervisor that controls computational resources
- Synchronization of both code and data
- Resource monitoring
- Full-functionally pause and continue on UI
- Auto control of the requirements
- Code dumping(with syntax highlight on UI)
- Kaggle integration
- Hierarchical logging
- Grid search
- Experiments comparison
- Customizing layouts


.. toctree::
   :caption: Overview:
   :maxdepth: 1

   self
   installation
   usage
   layout
   grid_search