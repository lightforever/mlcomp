from mlcomp.db.core import Session
from mlcomp.db.models import Dag
from mlcomp.db.providers import ModelProvider, DagProvider
from mlcomp.server.back.create_dags.standard import dag_standard
from mlcomp.utils.config import Config
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.misc import now


def dag_model_start(session: Session, data: dict):
    provider = ModelProvider(session)
    model = provider.by_id(data['model_id'])
    dag = DagProvider(session
                      ).by_id(data['dag'], joined_load=[Dag.project_rel])

    project = dag.project_rel
    src_config = Config.from_yaml(dag.config)
    pipe = src_config['pipes'][data['pipe']['name']]

    equations = yaml_load(model.equations)
    versions = data['pipe']['versions']
    if len(versions) > 0 and versions[0]['name'] == 'last':
        versions[0]['name'] = now().strftime('%Y.%m.%d %H:%M:%S')
    equations[data['pipe']['name']] = versions
    model.equations = yaml_dump(equations)

    if len(versions) > 0:
        equations = yaml_load(versions[0]['equations'])
        if len(pipe) == 1:
            pipe[list(pipe)[0]].update(equations)
        else:
            pipe.update(equations)

    for v in pipe.values():
        v['model_id'] = model.id

    config = {
        'info': {
            'name': data['pipe']['name'],
            'project': project.name
        },
        'executors': pipe
    }

    model.dag = data['dag']
    provider.commit()

    dag_standard(
        session=session,
        config=config,
        debug=False,
        upload_files=False,
        copy_files_from=data['dag']
    )


__all__ = ['dag_model_start']
