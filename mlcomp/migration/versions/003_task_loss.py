from sqlalchemy import Table, Column, MetaData, Float

meta = MetaData()


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn

        task = Table('task', meta, autoload=True)
        task_loss = Column('loss', Float)
        task_loss.create(task)
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
        task.c.loss.drop()
    except Exception:
        trans.rollback()
        raise
    else:
        trans.commit()
