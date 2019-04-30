from .base import *
from .task import Task

class Project(Base):
    __tablename__ = 'project'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    last_activity = sa.Column(sa.DateTime, default='Now()')
    dags = relationship("Dag")