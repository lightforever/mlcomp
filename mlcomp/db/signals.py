from functools import wraps

from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm.exc import StaleDataError

from mlcomp.db.providers import TaskProvider, StepProvider, DagProvider
from mlcomp.db.models import Task, Step, Log, ReportImg, File
from mlcomp.utils.misc import now
from mlcomp.db.core import Session
from sqlalchemy import event

_session = Session.create_session(key=__name__)


def error_handler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            if type(e) == ProgrammingError:
                Session.cleanup()
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


@event.listens_for(Step, 'after_insert')
@event.listens_for(Step, 'before_update')
@error_handler
def step_after_insert_update(mapper, connection, target):
    TaskProvider(_session).update_last_activity(target.task)


@event.listens_for(Log, 'after_insert')
@error_handler
def log_after_insert(mapper, connection, target):
    if target.step is None:
        return
    step = StepProvider(_session).by_id(target.step)
    TaskProvider(_session).update_last_activity(step.task)


@event.listens_for(ReportImg, 'after_insert')
def dag_after_create(mapper, connection, target):
    provider = DagProvider(_session)
    dag = provider.by_id(target.dag)
    dag.img_size += target.size
    provider.session.commit()


@event.listens_for(File, 'after_insert')
@error_handler
def file_after_create(mapper, connection, target):
    provider = DagProvider(_session)
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
