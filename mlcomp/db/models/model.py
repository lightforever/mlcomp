from mlcomp.db.models.base import *


class Model(Base):
    __tablename__ = 'model'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    score_local = sa.Column(sa.Float)
    score_public = sa.Column(sa.Float)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    created = sa.Column(sa.DateTime)
    file = sa.Column(sa.String)
    interface = sa.Column(sa.String)
    pred_file_valid = sa.Column(sa.String)
    pred_file_test = sa.Column(sa.String)
    interface_params = sa.Column(sa.String)
    slot = sa.Column(sa.String)

    dag_rel = relationship('Dag', lazy='noload')