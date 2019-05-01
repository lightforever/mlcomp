from .base import *
from mlcomp.db.enums import TaskStatus

class Task(Base):
    __tablename__ = 'task'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    status = sa.Column(sa.Integer)
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    last_activity = sa.Column(sa.DateTime)
    computer = sa.Column(sa.String, ForeignKey('computer.name'))
    gpu = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=1)
    executor = sa.Column(sa.String)
    status = sa.Column(sa.Integer, default=TaskStatus.NotRan.value)
    computer_assigned = sa.Column(sa.String, ForeignKey('computer.name'))
    memory = sa.Column(sa.Float, default=0.1)
    steps = sa.Column(sa.Integer, default=1)
    current_step = sa.Column(sa.Integer)
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    celery_id = sa.Column(sa.String)
    dag_rel = relationship('Dag', lazy='select')


class TaskDependence(Base):
    __tablename__ = 'task_dependencies'

    task_id = sa.Column(sa.Integer, primary_key=True)
    depend_id = sa.Column(sa.Integer, primary_key=True)