from mlcomp.db.core import Session
from mlcomp.db.models import Dag
from mlcomp.db.providers import ModelProvider, DagProvider
from mlcomp.server.back.create_dags.standard import dag_standard
from mlcomp.utils.config import Config
from mlcomp.utils.io import yaml_load, yaml_dump


def dag_model_start(session: Session, data: dict):
    provider = ModelProvider(session)
    model = provider.by_id(data['model_id'])
    dag = DagProvider(session
                      ).by_id(data['dag'], joined_load=[Dag.project_rel])

    project = dag.project_rel
    src_config = Config.from_yaml(dag.config)
    pipe = src_config['pipes'][data['pipe']['name']]
    config = {
        'info': {
            'name': data['pipe']['name'],
            'project': project.name
        },
        'executors': pipe
    }

    model.dag = data['dag']
    equations = yaml_load(model.equations)
    equations[data['pipe']['name']] = data['pipe']['equations']
    model.equations = yaml_dump(equations)
    provider.commit()

    dag_standard(
        session=session,
        config=config,
        debug=False,
        upload_files=False,
        copy_files_from=data['dag'],
        additional_info={
            'model_id': model.id,
            'equations': data['pipe']['equations']
        }
    )


__all__ = ['dag_model_start']
