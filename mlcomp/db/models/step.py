import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

from mlcomp.db.models.base import Base


class Step(Base):
    __tablename__ = 'step'

    id = sa.Column(sa.Integer, primary_key=True)
    level = sa.Column(sa.Integer)
    task = sa.Column(sa.Integer, ForeignKey('task.id'))
    started = sa.Column(sa.DateTime)
    finished = sa.Column(sa.DateTime)
    name = sa.Column(sa.String)
    task_rel = relationship('Task', lazy='noload')
    index = sa.Column(sa.Integer)


__all__ = ['Step']
