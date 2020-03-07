import sqlalchemy as sa

from mlcomp.db.models.base import Base


class Project(Base):
    __tablename__ = 'project'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String, nullable=False)
    class_names = sa.Column(sa.String, nullable=False)
    sync_folders = sa.Column(sa.String, nullable=False)
    ignore_folders = sa.Column(sa.String, nullable=False)


__all__ = ['Project']
