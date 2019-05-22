from .base import *

class Step(Base):
    __tablename__ = 'step'

    id = sa.Column(sa.Integer, primary_key=True)
    level = sa.Column(sa.Integer)
    task = sa.Column(sa.Integer, ForeignKey('task.id'))
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    status = sa.Column(sa.Integer)
    name = sa.Column(sa.String)
    task_rel = relationship('Task', lazy='noload')