import os
from os.path import join
import yaml
import shutil
from pathlib import Path

import torch

from catalyst.dl.scripts.trace import trace_model_from_checkpoint

from mlcomp.utils.settings import TASK_FOLDER, MODEL_FOLDER
from mlcomp.db.models import Model, Dag
from mlcomp.worker.executors.base import *
from mlcomp.db.providers import TaskProvider, ModelProvider, DagProvider
from mlcomp.utils.misc import now
from mlcomp.utils.config import Config


@Executor.register
class ModelAdd(Executor):
    __syn__ = 'model_add'

    def __init__(self,
                 name: str,
                 dag_pipe: int,
                 slot: str,
                 interface: str,
                 interface_params: dict,
                 train_task: int = None):
        self.dag_pipe = dag_pipe
        self.slot = slot
        self.interface = interface
        self.train_task = train_task
        self.name = name
        self.interface_params = interface_params

    def work(self):
        task = TaskProvider().by_id(self.train_task)
        dag = DagProvider().by_id(self.dag_pipe, joined_load=[Dag.project_rel])
        task_dir = join(TASK_FOLDER, str(task.id))
        src_log = f'{task_dir}/log'
        src_code = f'{src_log}/code'
        models_dir = join(MODEL_FOLDER, dag.project_rel.name)
        os.makedirs(models_dir, exist_ok=True)

        if not os.path.exists(src_code):
            os.symlink(task_dir, src_code, target_is_directory=True)

        model_path_tmp = f'{src_log}/traced.pth'
        traced = trace_model_from_checkpoint(Path(src_log), 'forward')

        model = Model(
            dag=self.dag_pipe,
            interface=self.interface,
            slot=self.slot,
            score_local=task.score,
            created=now(),
            name=self.name,
            project=dag.project,
            interface_params=yaml.dump(self.interface_params,
                                       default_flow_style=False)
        )
        provider = ModelProvider()
        provider.add(model)

        model_path = f'{models_dir}/{model.name}.pth'
        torch.jit.save(traced, model_path_tmp)
        shutil.copy(model_path_tmp, model_path)

        interface_params = yaml.load(model.interface_params)
        interface_params['file'] = model_path
        model.interface_params = yaml.dump(interface_params,
                                           default_flow_style=False)
        provider.update()

    @classmethod
    def _from_config(cls,
                     executor: dict,
                     config: Config,
                     additional_info: dict):
        return ModelAdd(
            name=executor['name'],
            dag_pipe=executor['dag'],
            slot=executor['slot'],
            interface=executor['interface'],
            train_task=executor['task'],
            interface_params=executor['interface_params']
        )


__all__ = ['ModelAdd']
