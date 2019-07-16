import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

from mlcomp.db.models.base import Base
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
    docker_img = sa.Column(sa.String)
    img_size = sa.Column(sa.BigInteger, nullable=False, default=0)
    file_size = sa.Column(sa.BigInteger, nullable=False, default=0)
    type = sa.Column(sa.Integer, default=0)


__all__ = ['Dag']
