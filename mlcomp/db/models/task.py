from .base import *

class Task(Base):
    __tablename__ = 'task'

    id = sa.Column(sa.Integer, primary_key=True)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    name = sa.Column(sa.String)
    parent_task = sa.Column(sa.Integer, ForeignKey('task.id'))
    parent_task_rel = relationship('Task')
    status = sa.Column(sa.Integer)
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    computer = sa.Column(sa.Integer)
    gpu = sa.Column(sa.Integer)
    cpu = sa.Column(sa.Integer)
    type = sa.Column(sa.Integer)
    status = sa.Column(sa.Integer)