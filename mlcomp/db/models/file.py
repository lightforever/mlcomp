import sys

import sqlalchemy as sa
from sqlalchemy import ForeignKey

from mlcomp.db.models.base import Base


class File(Base):
    __tablename__ = 'file'

    id = sa.Column(sa.Integer, primary_key=True)
    md5 = sa.Column(sa.String)
    created = sa.Column(sa.DateTime, default='Now()')
    content = sa.Column(sa.LargeBinary)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    size = sa.Column(sa.BigInteger, nullable=False, default=0)

    def __init__(self, **kwargs):
        super(File, self).__init__(**kwargs)
        self.size = sys.getsizeof(self.content)


__all__ = ['File']
