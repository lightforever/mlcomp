import os

from mlcomp.db.providers import *
from mlcomp.db.enums import DagType
from mlcomp.db.models import *
from mlcomp.worker.storage import Storage


def dag_pipe(config: dict, config_text: str = None):
    assert 'interfaces' in config, 'interfaces missed'
    assert 'pipes' in config, 'pipe missed'

    info = config['info']

    storage = Storage()
    dag_provider = DagProvider()

    folder = os.getcwd()
    project = ProjectProvider().by_name(info['project']).id
    dag = dag_provider.add(Dag(config=config_text,
                               project=project,
                               name=info['name'],
                               docker_img=info.get('docker_img'),
                               type=DagType.Pipe.value
                               ))
    storage.upload(folder, dag)

    # Change model dags which have the same name
    ModelProvider().change_dag(
        project=project,
        name=info['name'],
        to=dag.id
    )


__all__ = ['dag_pipe']