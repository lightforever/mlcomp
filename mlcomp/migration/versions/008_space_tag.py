from migrate import ForeignKeyConstraint
from sqlalchemy import Table, Column, MetaData, String

meta = MetaData()

table = Table(
    'space_tag', meta,
    Column('space', String(200), primary_key=True),
    Column('tag', String(100), primary_key=True),
)


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn
        table.create()

        space = Table('space', meta, autoload=True)
        ForeignKeyConstraint([table.c.space], [space.c.name],
                             ondelete='CASCADE').create()
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
