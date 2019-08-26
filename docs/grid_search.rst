Grid search
======================================

.. toctree::
   :maxdepth: 2

That is an example of a grid_search configuration:
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

The user needs to specify parameter sets in a grid field.

MLComp considers all possible combinations of them and transfers each combination to an executor within a special parameter: `additional_info`.

Each parameter set can be specified with 4 different ways:

1. key: list

    for example, batch_size: [20, 40, 80]

2. list of dicts

    for example,
    ::

        - - num_workers: 2
            lr: 0.01
          - num_workers: 3
            lr: 0.1

3. _folder: path_to_folder

    for example,

    ::

        _folder: configs/

    Each config in the folder must be a valid yml file.

    File structure is the same as in number 4.

4. _file: list of files

    for example,

    ::

        _file: [file1.yml]

    File structure is an ordinary dictionary. For example,

    ::

        num_workers: 2
        lr: 0.01
        stages:
            stage2:
                optimizer_params:
                  optimizer: Adam
                  lr: 0.001
                  weight_decay: 0.0001