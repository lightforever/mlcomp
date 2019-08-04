import sqlalchemy as sa

from mlcomp.db.models.base import Base


class Auxiliary(Base):
    __tablename__ = 'auxiliary'

    name = sa.Column(sa.String, primary_key=True)
    data = sa.Column(sa.String)


__all__ = ['Auxiliary']
