import sqlalchemy as sa
from sqlalchemy import ForeignKey

from mlcomp.db.models.base import Base


class Space(Base):
    __tablename__ = 'space'

    name = sa.Column(sa.String, nullable=False, primary_key=True)
    created = sa.Column(sa.DateTime, nullable=False)
    changed = sa.Column(sa.DateTime, nullable=False)
    content = sa.Column(sa.String, nullable=False)


class SpaceRelation(Base):
    __tablename__ = 'space_relation'

    parent = sa.Column(sa.String, ForeignKey('space.name'),
                       primary_key=True)
    child = sa.Column(sa.String, ForeignKey('space.name'),
                      primary_key=True)


__all__ = ['Space', 'SpaceRelation']
