from mlcomp.db.models.base import *


class Model(Base):
    __tablename__ = 'model'

    name = sa.Column(sa.String, primary_key=True)
    score_local = sa.Column(sa.Float)
    score_public = sa.Column(sa.Float)
    task = sa.Column(sa.Integer, ForeignKey('task.id'))
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    created = sa.Column(sa.DateTime)