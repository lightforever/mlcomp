import os

from mlcomp.db.core import Session
from mlcomp.db.enums import DagType
from mlcomp.db.providers import DagProvider, ProjectProvider, ModelProvider
from mlcomp.db.models import Dag
from mlcomp.worker.storage import Storage


def dag_pipe(session: Session, config: dict, config_text: str = None):
    assert 'pipes' in config, 'pipe missed'

    info = config['info']

    storage = Storage(session)
    dag_provider = DagProvider(session)

    folder = os.getcwd()
    project = ProjectProvider(session).by_name(info['project']).id
    dag = dag_provider.add(
        Dag(
            config=config_text,
            project=project,
            name=info['name'],
            docker_img=info.get('docker_img'),
            type=DagType.Pipe.value
        )
    )
    storage.upload(folder, dag)

    # Change model dags which have the same name
    ModelProvider(session
                  ).change_dag(project=project, name=info['name'], to=dag.id)


__all__ = ['dag_pipe']
