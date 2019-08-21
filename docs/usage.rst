Usage
======================================

.. toctree::
   :maxdepth: 2


Create yml configuration file with the following structure:

::

    info:
        name: Name of a DAG
        project: Name of your project
        layout: Name of your layout. Please, consider layout section
        expdir: root folder of your project
    executors:
      # declaring DAG structure
      executor_a:
        type: executor_a # name of your executor
      executor_b:
        type: exectuor_b
        depends: executor_a
      executor_c:
        type: executor_c
        depends: [executor_a, executor_b] # if your node depends on several components

MLComp has already had some very useful executors. For example, Catalyst. It can be used to train your deep neural networks.

In case of a desire to create your own, inherit the executor's class fom mlcomp.worker.executors.base.Executor.

Put it in any .py file, MLComp will use reflexion to find it.

::

    # MLComp will import a module that contains the class with the specified name
    # (register does not matter).
    @Executor.register
    class Executor_A(Executor):

        def work(self):
            # do some useful work
            pass

        @classmethod
        def _from_config(
            cls, executor: dict, config: Config, additional_info: dict
        ):
            # initialize your executor with the params you specified in the configuration file
            # they are available in the executor variable
            return cls(...)

Some service fields in an executor configuration are the following:

::

    gpu: 3 # you can specify requirements: gpu, cpu, memory(GB)
           # gpu can be set with a range; for example, 3-4
    cpu: 1
    memory: 0.1
    distr: True # use distributed training
    single_node: True # run only on a single work computer
    depends: either string or list # create a structure of your DAG
    grid: list of configurations # more details on a gird_search page