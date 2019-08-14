import os
from os.path import join
import shutil
from pathlib import Path

import safitty
import torch

from catalyst.dl.core import Experiment
from catalyst import utils
from catalyst.dl.utils.trace import trace_model
from catalyst.dl.utils.scripts import import_experiment_and_runner

from mlcomp import TASK_FOLDER, MODEL_FOLDER
from mlcomp.db.models import Model, Dag
from mlcomp.db.providers import TaskProvider, ModelProvider, DagProvider
from mlcomp.utils.misc import now
from mlcomp.utils.config import Config
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.worker.executors import Executor


@Executor.register
class ModelAdd(Executor):
    def __init__(
        self, name: str, dag_pipe: int, slot: str, interface: str,
        interface_params: dict, train_task: int, child_task: int
    ):
        self.dag_pipe = dag_pipe
        self.slot = slot
        self.interface = interface
        self.train_task = train_task
        self.name = name
        self.interface_params = interface_params
        self.child_task = child_task

    def trace_model_from_checkpoint(self, logdir, method_name):
        config_path = f'{logdir}/configs/_config.json'
        checkpoint_path = f'{logdir}/checkpoints/best.pth'
        self.info('Load config')
        config = safitty.load(config_path)
        if 'distributed_params' in config:
            del config['distributed_params']

        # Get expdir name
        # noinspection SpellCheckingInspection,PyTypeChecker
        # We will use copy of expdir from logs for reproducibility
        expdir_name = os.path.basename(config['args']['expdir'])
        expdir_from_logs = f'{logdir}/code/{expdir_name}'

        self.info('Import experiment and runner from logdir')
        ExperimentType, RunnerType = \
            import_experiment_and_runner(Path(expdir_from_logs))
        experiment: Experiment = ExperimentType(config)

        self.info('Load model state from checkpoints/best.pth')
        model = experiment.get_model(next(iter(experiment.stages)))
        checkpoint = utils.load_checkpoint(checkpoint_path)
        utils.unpack_checkpoint(checkpoint, model=model)

        self.info('Tracing')
        traced = trace_model(model, experiment, RunnerType, method_name)

        self.info('Done')
        return traced

    def work(self):
        task_provider = TaskProvider(self.session)
        task = task_provider.by_id(self.train_task)
        dag = DagProvider(
            self.session
        ).by_id(self.dag_pipe, joined_load=[Dag.project_rel])

        task_dir = join(TASK_FOLDER, str(self.child_task or task.id))
        src_log = f'{task_dir}/log'
        src_code = f'{src_log}/code'
        models_dir = join(MODEL_FOLDER, dag.project_rel.name)
        os.makedirs(models_dir, exist_ok=True)

        self.info(f'Task = {self.task} child_task: {self.child_task}')

        if not os.path.exists(src_code):
            os.symlink(task_dir, src_code, target_is_directory=True)

        model_path_tmp = f'{src_log}/traced.pth'
        traced = self.trace_model_from_checkpoint(src_log, 'forward')

        model = Model(
            dag=self.dag_pipe,
            interface=self.interface,
            slot=self.slot,
            score_local=task.score,
            created=now(),
            name=self.name,
            project=dag.project,
            interface_params=yaml_dump(self.interface_params)
        )
        provider = ModelProvider(self.session)
        provider.add(model, commit=False)
        try:
            model_path = f'{models_dir}/{model.name}.pth'
            model_weight_path = f'{models_dir}/{model.name}_weight.pth'
            torch.jit.save(traced, model_path_tmp)
            shutil.copy(model_path_tmp, model_path)
            shutil.copy(f'{src_log}/checkpoints/best.pth', model_weight_path)

            interface_params = yaml_load(model.interface_params)
            interface_params['file'] = model_path
            model.interface_params = yaml_dump(interface_params)
            provider.update()
        except Exception as e:
            provider.rollback()
            raise e

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
        return ModelAdd(
            name=executor['name'],
            dag_pipe=executor['dag'],
            slot=executor['slot'],
            interface=executor['interface'],
            train_task=executor['task'],
            interface_params=executor['interface_params'],
            child_task=executor['child_task']
        )


__all__ = ['ModelAdd']
