from sqlalchemy import Table, Column, MetaData, String

meta = MetaData()


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn

        table = Table('project', meta, autoload=True)
        col = Column('sync_folders', String(8000))
        col.create(table)
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

        table = Table('project', meta, autoload=True)
        table.c.continued.drop()
    except Exception:
        trans.rollback()
        raise
    else:
        trans.commit()
