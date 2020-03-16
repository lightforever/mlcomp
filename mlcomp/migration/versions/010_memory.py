from sqlalchemy import Table, Column, MetaData, String, Integer, Float

meta = MetaData()

table = Table(
    'memory', meta,
    Column('id', Integer, primary_key=True),
    Column('model', String(200), nullable=False),
    Column('memory', Float, nullable=False),
    Column('batch_size', Integer, nullable=False),
    Column('num_classes', Integer),
    Column('variant', String(200)),
    Column('img_size', Integer),
)


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn
        table.create()
    except Exception:
        trans.rollback()
        raise
    else:
        trans.commit()


def downgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn
        table.drop()
    except Exception:
        trans.rollback()
        raise
    else:
        trans.commit()
