from .base import *

class TaskStorage(Base):
    __tablename__ = 'task_storage'

    id = sa.Column(sa.Integer, primary_key=True)
    task = sa.Column(sa.Integer, ForeignKey('task.id'))
    file = sa.Column(sa.Integer, ForeignKey('file.id'))
    path = sa.Column(sa.String)
    is_dir = sa.Column(sa.Boolean)