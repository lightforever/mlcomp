from mlcomp.db.core import Session
from mlcomp.db.models import Dag
from mlcomp.db.providers import ModelProvider, DagProvider
from mlcomp.server.back.create_dags.standard import dag_standard
from mlcomp.utils.config import Config


def dag_model_start(session: Session, data: dict):
    provider = ModelProvider(session)
    model = provider.by_id(data['model_id'])
    dag = DagProvider(session
                      ).by_id(data['dag'], joined_load=[Dag.project_rel])

    project = dag.project_rel
    src_config = Config.from_yaml(dag.config)
    pipe = src_config['pipes'][data['pipe']]
    config = {
        'info': {
            'name': data['pipe'],
            'project': project.name
        },
        'executors': pipe
    }

    dag_standard(
        session=session,
        config=config,
        debug=False,
        upload_files=False,
        copy_files_from=data['dag'],
        additional_info={'model_id': model.id, 'equations': data['equations']}
    )

    model.dag = data['dag']
    model.equations = data['equations']
    provider.commit()


__all__ = ['dag_model_start']
