import shutil
import os

from mlcomp.db.models import Model
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
                 train_task: int = None):
        self.dag_pipe = dag_pipe
        self.slot = slot
        self.interface = interface
        self.train_task = train_task
        self.name = name

    def work(self):
        task = TaskProvider().by_id(self.train_task)
        dag = DagProvider().by_id(self.dag_pipe)
        src = f'/opt/mlcomp/tasks/{task.id}/log/checkpoints/best.pth'
        dst = f'/opt/mlcomp/models/{task.name}_{task.id}.pth'

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        model = Model(
            dag=self.dag_pipe,
            interface=self.interface,
            slot=self.slot,
            file=dst,
            score_local=task.score,
            created=now(),
            name=self.name,
            project=dag.project
        )
        ModelProvider().add(model)

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
            train_task=executor['task']
        )


__all__ = ['ModelAdd']
