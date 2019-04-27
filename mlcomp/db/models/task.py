from .base import *
from mlcomp.db.enums import TaskStatus

class Task(Base):
    __tablename__ = 'task'

    id = sa.Column(sa.Integer, primary_key=True)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    name = sa.Column(sa.String)
    status = sa.Column(sa.Integer)
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    computer = sa.Column(sa.Integer)
    gpu = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=0)
    executor = sa.Column(sa.String)
    status = sa.Column(sa.Integer, default=TaskStatus.NotRan.value)
    config = sa.Column(sa.String)

class TaskDependence(Base):
    __tablename__ = 'task_dependencies'

    task_id = sa.Column(sa.Integer, ForeignKey('task.id'), primary_key=True)
    depend_id = sa.Column(sa.Integer, ForeignKey('task.id'), primary_key=True)