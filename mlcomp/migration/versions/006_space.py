from sqlalchemy import Table, Column, MetaData, String, TIMESTAMP

meta = MetaData()


table = Table(
    'space', meta,
    Column('name', String(200), primary_key=True),
    Column('changed', TIMESTAMP, nullable=False),
    Column('created', TIMESTAMP, nullable=False),
    Column('content', String(10000), nullable=False),
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
