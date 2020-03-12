import sqlalchemy as sa

from mlcomp.db.models.base import Base


class Memory(Base):
    __tablename__ = 'memory'

    id = sa.Column(sa.Integer, primary_key=True)
    model = sa.Column(sa.String, nullable=False)
    variant = sa.Column(sa.String)
    num_classes = sa.Column(sa.Integer)
    img_size = sa.Column(sa.Integer)
    batch_size = sa.Column(sa.Integer, nullable=False)
    memory = sa.Column(sa.Float, nullable=False)


__all__ = ['Memory']
