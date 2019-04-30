from .base import *
from mlcomp.db.enums import DagStatus
from mlcomp.utils.misc import now

class Dag(Base):
    __tablename__ = 'dag'

    id = sa.Column(sa.Integer, primary_key=True)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    created = sa.Column(sa.DateTime, default=now())
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    config = sa.Column(sa.String)
    name = sa.Column(sa.String)
    status = sa.Column(sa.Integer, default=DagStatus.NotRan.value)
