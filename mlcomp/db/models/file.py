from .base import *

class File(Base):
    __tablename__ = 'file'

    id = sa.Column(sa.Integer, primary_key=True)
    md5 = sa.Column(sa.String)
    created = sa.Column(sa.DateTime, default='Now()')
    content = sa.Column(sa.LargeBinary)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
