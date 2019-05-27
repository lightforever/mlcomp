from mlcomp.utils.misc import now
from .base import *

class Computer(Base):
    __tablename__ = 'computer'

    name = sa.Column(sa.String, primary_key=True)
    gpu = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=1)
    memory = sa.Column(sa.Float, default=0.1)
    usage = sa.Column(sa.String)

class ComputerUsage(Base):
    __tablename__ = 'computer_usage'

    id = sa.Column(sa.Integer, primary_key=True)
    computer = sa.Column(sa.String, ForeignKey('computer.name'))
    usage = sa.Column(sa.String)
    time = sa.Column(sa.DateTime, default=now())