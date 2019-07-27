import sqlalchemy as sa
from sqlalchemy import ForeignKey

from mlcomp.db.models.base import Base


class Docker(Base):
    __tablename__ = 'docker'

    name = sa.Column(sa.String, primary_key=True)
    computer = sa.Column(sa.String,
                         ForeignKey('computer.name'),
                         primary_key=True)
    last_activity = sa.Column(sa.DateTime, nullable=False)
    ports = sa.Column(sa.String, nullable=False)


__all__ = ['Docker']
