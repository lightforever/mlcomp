from .base import *
from mlcomp.utils.misc import now

class Dag(Base):
    __tablename__ = 'dag'

    id = sa.Column(sa.Integer, primary_key=True)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    created = sa.Column(sa.DateTime, default=now())
    config = sa.Column(sa.String)
    name = sa.Column(sa.String)
    tasks = relationship('Task', lazy='noload')
    project_rel = relationship('Project', lazy='noload')

