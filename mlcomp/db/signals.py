from mlcomp.db.providers import TaskProvider, StepProvider, DagProvider
from mlcomp.db.models import Task, Step, Log, ReportImg, File
from mlcomp.utils.misc import now
from mlcomp.db.core import Session
from sqlalchemy import event

signals_session = Session.create_session(key='signals')


@event.listens_for(Task, 'before_update')
def task_before_update(mapper, connection, target):
    target.last_activity = now()


@event.listens_for(Step, 'after_insert')
@event.listens_for(Step, 'before_update')
def step_after_insert_update(mapper, connection, target):
    TaskProvider(signals_session).update_last_activity(target.task)


@event.listens_for(Log, 'after_insert')
def log_after_insert(mapper, connection, target):
    if target.step is None:
        return
    step = StepProvider().by_id(target.step)
    TaskProvider(signals_session).update_last_activity(step.task)


@event.listens_for(ReportImg, 'after_insert')
def dag_after_create(mapper, connection, target):
    provider = DagProvider(signals_session)
    dag = provider.by_id(target.dag)
    dag.img_size += target.size
    provider.session.commit()


@event.listens_for(File, 'after_insert')
def file_after_create(mapper, connection, target):
    provider = DagProvider(signals_session)
    dag = provider.by_id(target.dag)
    dag.file_size += target.size
    provider.session.commit()


__all__ = [
    'task_before_update',
    'step_after_insert_update',
    'log_after_insert',
    'dag_after_create',
    'file_after_create'
]
