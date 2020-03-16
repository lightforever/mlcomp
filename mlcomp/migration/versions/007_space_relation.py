from migrate import ForeignKeyConstraint
from sqlalchemy import Table, Column, MetaData, String

meta = MetaData()

table = Table(
    'space_relation', meta,
    Column('parent', String(200), primary_key=True),
    Column('child', String(200), primary_key=True),
)


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn
        table.create()

        space = Table('space', meta, autoload=True)
        ForeignKeyConstraint([table.c.parent], [space.c.name],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([table.c.child], [space.c.name],
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
