import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, deferred

from mlcomp.db.enums import TaskStatus
from mlcomp.db.models.base import Base


class Task(Base):
    __tablename__ = 'task'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    last_activity = sa.Column(sa.DateTime)
    computer = sa.Column(sa.String)
    gpu = sa.Column(sa.Integer, default=0)
    gpu_max = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=1)
    executor = sa.Column(sa.String)
    status = sa.Column(sa.Integer, default=TaskStatus.NotRan.value)
    computer_assigned = sa.Column(sa.String, ForeignKey('computer.name'))
    computer_assigned_rel = relationship('Computer', lazy='noload')

    memory = sa.Column(sa.Float, default=0.1)
    steps = sa.Column(sa.Integer, default=1)
    current_step = sa.Column(sa.String)
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    celery_id = sa.Column(sa.String)
    dag_rel = relationship('Dag', lazy='noload')
    debug = sa.Column(sa.Boolean, default=False)
    pid = sa.Column(sa.Integer)
    worker_index = sa.Column(sa.Integer)
    docker_assigned = sa.Column(sa.String)
    type = sa.Column(sa.Integer)
    score = sa.Column(sa.Float)
    report = sa.Column(sa.Integer, ForeignKey('report.id'))
    report_rel = relationship('Report', lazy='noload')
    gpu_assigned = sa.Column(sa.String)
    parent = sa.Column(sa.Integer, ForeignKey('task.id'))
    parent_rel = relationship('Task', lazy='noload')
    loss = sa.Column(sa.Float)

    batch_index = sa.Column(sa.Integer)
    batch_total = sa.Column(sa.Integer)
    loader_name = sa.Column(sa.String)
    epoch_duration = sa.Column(sa.Integer)
    epoch_time_remaining = sa.Column(sa.Integer)

    result = deferred(sa.Column(sa.String))
    additional_info = deferred(sa.Column(sa.String))


class TaskDependence(Base):
    __tablename__ = 'task_dependency'

    task_id = sa.Column(sa.Integer, primary_key=True)
    depend_id = sa.Column(sa.Integer, primary_key=True)


class TaskSynced(Base):
    __tablename__ = 'task_synced'

    computer = sa.Column(sa.String, ForeignKey('computer.name'),
                         primary_key=True)
    task = sa.Column(sa.Integer, ForeignKey('task.id'), primary_key=True)


__all__ = ['Task', 'TaskDependence', 'TaskSynced']
