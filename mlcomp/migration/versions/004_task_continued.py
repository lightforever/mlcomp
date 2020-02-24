from sqlalchemy import Table, Column, MetaData, Boolean

meta = MetaData()


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = migrate_engine

        task = Table('task', meta, autoload=True)
        col = Column('continued', Boolean,  default=False)
        col.create(task)
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

        task = Table('task', meta, autoload=True)
        task.c.continued.drop()
    except Exception:
        trans.rollback()
        raise
    else:
        trans.commit()
