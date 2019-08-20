Grid search
======================================

.. toctree::
   :maxdepth: 2

An example of a grid_search configuration:
::

    executors:
      train:
        type: catalyst
        args:
          config: catalyst.yml
        grid:
          - batch_size: [20, 40, 80]
          - - num_workers: 2
              lr: 0.01
            - num_workers: 3
              lr: 0.1

In a grid you need to specify parameter sets.

MLComp will use all possible combinations of them and transfer each combination to an executor in a special parameter: `additional_info`.

Each parameter set can be specified with 4 different ways:

1. key: list

    an example: batch_size: [20, 40, 80]

2. list of dicts

    and example:
    ::

        - - num_workers: 2
            lr: 0.01
          - num_workers: 3
            lr: 0.1

3. _folder: path_to_folder

    an example:

    ::

        _folder: configs/

    Each config in the folder must be a valid yml file.

    File structure: the same as in 4.

4. _file: list of files

    an example:

    ::

        _file: [file1.yml]

    File structure: an ordinary dictionary, an example:

    ::

        num_workers: 2
        lr: 0.01
        stages:
            stage2:
                optimizer_params:
                  optimizer: Adam
                  lr: 0.001
                  weight_decay: 0.0001