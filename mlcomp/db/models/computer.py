from .base import *

class Computer(Base):
    __tablename__ = 'computer'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    gpu = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=1)
    memory = sa.Column(sa.Float, default=0.1)
