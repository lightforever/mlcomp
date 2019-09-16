import sqlalchemy as sa
from sqlalchemy import ForeignKey

from mlcomp.db.models.base import Base
from mlcomp.utils.misc import now


class Computer(Base):
    __tablename__ = 'computer'

    name = sa.Column(sa.String, primary_key=True)
    gpu = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=1)
    memory = sa.Column(sa.Float, default=0.1)
    usage = sa.Column(sa.String)
    ip = sa.Column(sa.String)
    port = sa.Column(sa.Integer)
    user = sa.Column(sa.String)
    last_synced = sa.Column(sa.DateTime)
    disk = sa.Column(sa.Integer)
    syncing_computer = sa.Column(sa.String, ForeignKey('computer.name'))
    root_folder = sa.Column(sa.String)
    can_process_tasks = sa.Column(sa.Boolean)
    sync_with_this_computer = sa.Column(sa.Boolean)


class ComputerUsage(Base):
    __tablename__ = 'computer_usage'

    id = sa.Column(sa.Integer, primary_key=True)
    computer = sa.Column(sa.String, ForeignKey('computer.name'))
    usage = sa.Column(sa.String)
    time = sa.Column(sa.DateTime, default=now())


__all__ = ['Computer', 'ComputerUsage']
