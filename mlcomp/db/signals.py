from mlcomp.db.providers import *
from sqlalchemy import event


signals_session = Session.create_session(key='signals')

@event.listens_for(Task, 'before_update')
def task_before_update(mapper, connection, target):
    target.last_activity = now()

@event.listens_for(Step, 'after_insert')
@event.listens_for(Step, 'after_update')
def step_after_insert_update(mapper, connection, target):
    TaskProvider(signals_session).update_last_activity(target.task)


@event.listens_for(Log, 'after_insert')
def log_after_insert(mapper, connection, target):
    step = StepProvider(signals_session).by_id(target.step)
    TaskProvider(signals_session).update_last_activity(step.task)
