from .base import *

class Computer(Base):
    __tablename__ = 'computer'

    name = sa.Column(sa.String, primary_key=True)
    gpu = sa.Column(sa.Integer, default=0)
    cpu = sa.Column(sa.Integer, default=1)
    memory = sa.Column(sa.Float, default=0.1)
