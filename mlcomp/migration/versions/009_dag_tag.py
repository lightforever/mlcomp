from migrate import ForeignKeyConstraint
from sqlalchemy import Table, Column, MetaData, String, Integer

meta = MetaData()

table = Table(
    'dag_tag', meta,
    Column('dag', Integer, primary_key=True),
    Column('tag', String(100), primary_key=True),
)


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn
        table.create()

        dag = Table('dag', meta, autoload=True)
        ForeignKeyConstraint([table.c.dag], [dag.c.id],
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
