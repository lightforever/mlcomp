from functools import wraps

from sqlalchemy import event
from sqlalchemy.orm.exc import StaleDataError

from mlcomp.db.providers import TaskProvider, StepProvider, DagProvider
from mlcomp.db.models import Task, Step, Log, ReportImg, File
from mlcomp.utils.misc import now
from mlcomp.db.core import Session

_session = Session.create_session(key=__name__)


def error_handler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        global _session
        try:
            f(*args, **kwargs)
        except Exception as e:
            if Session.sqlalchemy_error(e):
                Session.cleanup(key=__name__)
                _session = Session.create_session(key=__name__)
            raise e

    return decorated


@event.listens_for(Task, 'before_update')
@error_handler
def task_before_update(mapper, connection, target):
    target.last_activity = now()
    if target.parent:
        provider = TaskProvider(_session)
        parent = provider.by_id(target.parent)
        if parent is None:
            return

        parent.last_activity = target.last_activity

        try:
            provider.commit()
        except StaleDataError:
            pass


@event.listens_for(Step, 'before_insert')
@event.listens_for(Step, 'before_update')
@error_handler
def step_before_insert_update(mapper, connection, target):
    TaskProvider(_session).update_last_activity(target.task)


@event.listens_for(Log, 'before_insert')
@error_handler
def log_before_insert(mapper, connection, target):
    if target.step is None:
        return
    step = StepProvider(_session).by_id(target.step)
    TaskProvider(_session).update_last_activity(step.task)


@event.listens_for(ReportImg, 'before_insert')
def dag_before_create(mapper, connection, target):
    provider = DagProvider(_session)
    dag = provider.by_id(target.dag)
    dag.img_size += target.size
    provider.commit()


@event.listens_for(File, 'before_insert')
@error_handler
def file_before_create(mapper, connection, target):
    provider = DagProvider(_session)

    dag = provider.by_id(target.dag)
    dag.file_size += target.size
    provider.commit()


__all__ = [
    'task_before_update', 'step_before_insert_update', 'log_before_insert',
    'dag_before_create', 'file_before_create'
]
