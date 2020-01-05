from sqlalchemy import Table, Column, MetaData, BigInteger

meta = MetaData()


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn

        task = Table('project', meta, autoload=True)
        task_file_size = Column('file_size', BigInteger)
        task_file_size.create(task)
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
        task.c.file_size.drop()
    except Exception:
        trans.rollback()
        raise
    else:
        trans.commit()
