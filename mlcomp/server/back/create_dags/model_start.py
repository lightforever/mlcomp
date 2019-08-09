from mlcomp.db.core import Session
from mlcomp.db.models import Dag
from mlcomp.db.providers import ModelProvider, DagProvider
from mlcomp.server.back.create_dags.standard import dag_standard
from mlcomp.utils.config import Config
from mlcomp.utils.io import yaml_load


def dag_model_start(session: Session, data: dict):
    provider = ModelProvider(session)
    model = provider.by_id(data['model_id'])
    dag = DagProvider(session
                      ).by_id(data['dag'], joined_load=[Dag.project_rel])

    project = dag.project_rel
    src_config = Config.from_yaml(dag.config)
    pipe = src_config['pipes'][data['pipe']]
    for k, v in pipe.items():
        if v.get('slot') != data['slot']:
            continue
        params = yaml_load(data['interface_params'])
        slot = {
            'interface': data['interface'],
            'interface_params': params,
            'slot': k,
            'name': model.name,
            'id': data['model_id']
        }
        v['slot'] = slot

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
        copy_files_from=data['dag']
    )

    model.dag = data['dag']
    model.interface = data['interface']
    model.interface_params = data['interface_params']
    model.slot = data['slot']

    provider.commit()


__all__ = ['dag_model_start']
