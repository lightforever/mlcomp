from .base import *

class DagStorage(Base):
    __tablename__ = 'dag_storage'

    id = sa.Column(sa.Integer, primary_key=True)
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    file = sa.Column(sa.Integer, ForeignKey('file.id'))
    path = sa.Column(sa.String)
    is_dir = sa.Column(sa.Boolean)

class DagLibrary(Base):
    __tablename__ = 'dag_library'

    id = sa.Column(sa.Integer, primary_key=True)
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    library = sa.Column(sa.String)
    version = sa.Column(sa.String)