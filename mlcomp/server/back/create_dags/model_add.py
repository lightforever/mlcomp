from sqlalchemy.orm import joinedload

from mlcomp.db.core import Session
from mlcomp.db.models import Task
from mlcomp.db.providers import TaskProvider, ProjectProvider
from mlcomp.server.back.create_dags.standard import dag_standard
from mlcomp.utils.io import yaml_load


def dag_model_add(session: Session, data: dict):
    task_provider = TaskProvider(session)
    task = task_provider.by_id(
        data['task'], options=joinedload(Task.dag_rel, innerjoin=True)
    )
    child_tasks = task_provider.children(task.id)
    computer = task.computer_assigned
    child_task = None
    if len(child_tasks) > 0:
        child_task = child_tasks[0].id
        computer = child_tasks[0].computer_assigned

    project = ProjectProvider(session).by_id(task.dag_rel.project)
    interface_params = data.get('interface_params', '')
    interface_params = yaml_load(interface_params)
    config = {
        'info': {
            'name': 'model_add',
            'project': project.name,
            'computer': computer
        },
        'executors': {
            'model_add': {
                'type': 'model_add',
                'dag': data['dag'],
                'slot': data['slot'],
                'interface': data['interface'],
                'task': data.get('task'),
                'name': data['name'],
                'interface_params': interface_params,
                'child_task': child_task
            }
        }
    }

    dag_standard(
        session=session, config=config, debug=False, upload_files=False
    )


__all__ = ['dag_model_add']
