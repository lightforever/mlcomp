from sqlalchemy import Table, Column, Integer, String, MetaData, Float, TIMESTAMP, Boolean, LargeBinary, Index
from migrate.changeset.constraint import ForeignKeyConstraint, UniqueConstraint

meta = MetaData()

computer = Table(
    'computer', meta,
    Column('name', String(100), primary_key=True),
    Column('gpu', Integer, default=0, nullable=False),
    Column('cpu', Integer, default=1, nullable=False),
    Column('memory', Float, default=0.1, nullable=False),
    Column('usage', String(250))
)

computer_usage = Table(
    'computer_usage', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('computer', String(100), nullable=False),
    Column('usage', String(500), nullable=False),
    Column('time', TIMESTAMP, nullable=False, default='now()')
)

dag = Table(
    'dag', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(100), nullable=False),
    Column('created', TIMESTAMP, nullable=False, default='now()'),
    Column('config', String(8000), nullable=False),
    Column('project', Integer, nullable=False),
    Column('docker_img', String(100), nullable=True),
)

dag_library = Table(
    'dag_library', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('library', String(100), nullable=False),
    Column('version', String(30), nullable=False),
    Column('dag', Integer, nullable=False)
)

dag_storage = Table(
    'dag_storage', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('dag', Integer, nullable=False),
    Column('file', Integer),
    Column('path', String(210), nullable=False),
    Column('is_dir', Boolean, nullable=False, default=False)
)

file = Table(
    'file', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('md5', String(32), nullable=False),
    Column('created', TIMESTAMP, nullable=False, default='now()'),
    Column('content', LargeBinary, nullable=False),
    Column('project', Integer, nullable=False),
    Column('dag', Integer, nullable=False)
)

log = Table(
    'log', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('step', Integer),
    Column('message', String(4000), nullable=False),
    Column('time', TIMESTAMP, nullable=False, default='now()'),
    Column('level', Integer, nullable=False),
    Column('component', Integer, nullable=False),
)

project = Table(
    'project', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(180), nullable=False)
)

report = Table(
    'report', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('config', String(4000), nullable=False),
    Column('time', TIMESTAMP, nullable=False, default='now()'),
    Column('name', String(100), nullable=False),
    Column('project', Integer, nullable=False)
)

report_img = Table(
    'report_img', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('epoch', Integer, nullable=False),
    Column('task', Integer, nullable=False),
    Column('group', String(100), nullable=False),
    Column('img', LargeBinary, nullable=False),
    Column('number', Integer, nullable=False),
    Column('project', Integer, nullable=False),
    Column('dag', Integer, nullable=False),
)

report_series = Table(
    'report_series', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('task', Integer, nullable=False),
    Column('time', TIMESTAMP, nullable=False, default='now()'),
    Column('epoch', Integer, nullable=False),
    Column('value', Float, nullable=False),
    Column('name', String(100), nullable=False),
    Column('group', String(50), nullable=False),
)

report_task = Table(
    'report_task', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('report', Integer, nullable=False),
    Column('task', Integer, nullable=False),
)

step = Table(
    'step', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('task', Integer, nullable=False),
    Column('level', Integer, nullable=False),
    Column('started', TIMESTAMP, nullable=False),
    Column('finished', TIMESTAMP),
    Column('status', Integer, nullable=False),
    Column('name', String(100), nullable=False)
)

task = Table(
    'task', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(180), nullable=False),
    Column('status', Integer, nullable=False),
    Column('started', TIMESTAMP),
    Column('finished', TIMESTAMP),
    Column('computer', String(100)),
    Column('gpu', Integer, nullable=False, default=0),
    Column('cpu', Integer, nullable=False, default=1),
    Column('executor', String(100), nullable=False),
    Column('computer_assigned', String(100)),
    Column('memory', Float, nullable=False, default=0.1),
    Column('steps', Integer, nullable=False, default=1),
    Column('current_step', Integer),
    Column('dag', Integer, nullable=False),
    Column('celery_id', String(50)),
    Column('last_activity', TIMESTAMP),
    Column('debug', Boolean, nullable=False, default=False),
    Column('pid', Integer)
)

task_dependency = Table(
    'task_dependency', meta,
    Column('task_id', Integer, primary_key=True),
    Column('depend_id', Integer, primary_key=True),
)


def upgrade(migrate_engine):
    meta.bind = migrate_engine

    computer.create()
    computer_usage.create()
    dag.create()
    dag_library.create()
    dag_storage.create()
    file.create()
    log.create()
    project.create()
    report.create()
    report_img.create()
    report_series.create()
    report_task.create()
    step.create()
    task.create()
    task_dependency.create()

    ForeignKeyConstraint([computer_usage.c.computer], [computer.c.name], ondelete='CASCADE').create()
    Index('computer_name_idx', computer.c.name).create()

    Index('computer_usage_time_idx', computer_usage.c.time.desc()).create()
    Index('computer_usage_id_idx', computer_usage.c.id.desc()).create()

    ForeignKeyConstraint([dag.c.project], [project.c.id], ondelete='CASCADE').create()
    Index('dag_project_idx', dag.c.project.desc()).create()
    Index('dag_created_idx', dag.c.created.desc()).create()
    ForeignKeyConstraint([dag_library.c.dag], [dag.c.id], ondelete='CASCADE').create()
    Index('dag_library_task_idx', dag_library.c.dag.desc()).create()
    Index('dag_id_idx', dag.c.id.desc()).create()

    ForeignKeyConstraint([dag_storage.c.dag], [dag.c.id], ondelete='CASCADE').create()
    ForeignKeyConstraint([dag_storage.c.file], [file.c.id], ondelete='CASCADE').create()
    Index('dag_storage_dag_idx', dag_storage.c.dag.desc()).create()
    Index('dag_storage_id_idx', dag_storage.c.id.desc()).create()

    ForeignKeyConstraint([file.c.project], [project.c.id], ondelete='CASCADE').create()
    Index('file_created_idx', file.c.created.desc()).create()
    Index('file_project_idx', file.c.project.desc()).create()
    UniqueConstraint(file.c.md5, name='file_md5_idx').create()
    Index('file_id_idx', file.c.id.desc()).create()

    ForeignKeyConstraint([log.c.step], [step.c.id], ondelete='CASCADE').create()
    Index('log_step_idx', log.c.step.desc()).create()
    Index('log_time_idx', log.c.time.desc()).create()
    Index('log_id_idx', log.c.id.desc()).create()

    Index('project_id_idx', project.c.id.desc()).create()
    UniqueConstraint(project.c.name, name='project_name').create()

    ForeignKeyConstraint([report.c.project], [project.c.id], ondelete='CASCADE').create()
    Index('report_id_idx', report.c.id.desc()).create()

    ForeignKeyConstraint([report_img.c.project], [task.c.id], ondelete='CASCADE').create()
    Index('report_img_project_idx', report_img.c.project.desc()).create()
    Index('report_img_task_idx', report_img.c.task.desc()).create()
    Index('report_img_id_idx', report_img.c.id.desc()).create()

    ForeignKeyConstraint([report_series.c.task], [task.c.id], ondelete='CASCADE').create()
    Index('report_series_id_idx', report_series.c.id.desc()).create()

    ForeignKeyConstraint([report_task.c.task], [task.c.id], ondelete='CASCADE').create()
    ForeignKeyConstraint([report_task.c.report], [report.c.id], ondelete='CASCADE').create()
    Index('report_task_id_idx', report_task.c.id.desc()).create()

    ForeignKeyConstraint([step.c.task], [task.c.id], ondelete='CASCADE').create()
    Index('step_name_idx', step.c.name).create()
    Index('step_id_idx', step.c.id.desc()).create()

    ForeignKeyConstraint([task.c.computer], [computer.c.name], ondelete='CASCADE').create()
    ForeignKeyConstraint([task.c.computer_assigned], [computer.c.name], ondelete='CASCADE').create()
    ForeignKeyConstraint([task.c.dag], [dag.c.id], ondelete='CASCADE').create()

    Index('task_dag_idx', task.c.dag.desc()).create()
    Index('task_finished_idx', task.c.finished.desc()).create()
    Index('task_name_idx', task.c.name).create()
    Index('task_started_idx', task.c.started.desc()).create()

    Index('task_id_idx', task.c.id.desc()).create()
    Index('task_dependency_task_idx', task_dependency.c.task_id.desc()).create()
    Index('task_dependency_depend_idx', task_dependency.c.depend_id.desc()).create()

    ForeignKeyConstraint([task_dependency.c.task_id], [task.c.id], ondelete='CASCADE').create()
    ForeignKeyConstraint([task_dependency.c.depend_id], [task.c.id], ondelete='CASCADE').create()

def downgrade(migrate_engine):
    meta.bind = migrate_engine

    computer.drop()
    computer_usage.drop()
    dag.drop()
    dag_library.drop()
    dag_storage.drop()
    file.drop()
    log.drop()
    project.drop()
    report.drop()
    report_img.drop()
    report_series.drop()
    report_task.drop()
    step.drop()
    task.drop()
    task_dependency.drop()
