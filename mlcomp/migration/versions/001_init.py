from sqlalchemy import Table, Column, Integer, String, MetaData, Float, \
    TIMESTAMP, Boolean, LargeBinary, Index, BigInteger
from migrate.changeset.constraint import ForeignKeyConstraint, UniqueConstraint

meta = MetaData()

auxiliary = Table(
    'auxiliary', meta,
    Column('name', String(100), primary_key=True),
    Column('data', String(16000), nullable=False)
)

computer = Table(
    'computer', meta,
    Column('name', String(100), primary_key=True),
    Column('gpu', Integer, default=0, nullable=False),
    Column('cpu', Integer, default=1, nullable=False),
    Column('memory', Float, default=0.1, nullable=False),
    Column('usage', String(2000)),
    Column('ip', String(100), nullable=False),
    Column('port', Integer, nullable=False),
    Column('user', String, nullable=False),
    Column('last_synced', TIMESTAMP),
    Column('disk', Integer, nullable=False),
    Column('syncing_computer', String(100))

)

computer_usage = Table(
    'computer_usage', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('computer', String(100), nullable=False),
    Column('usage', String(4000), nullable=False),
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
    Column('img_size', BigInteger, nullable=False),
    Column('file_size', BigInteger, nullable=False),
    Column('type', Integer, nullable=False),
    Column('report', Integer)
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
    Column('dag', Integer, nullable=False),
    Column('size', BigInteger, nullable=False),
)

log = Table(
    'log', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('step', Integer),
    Column('message', String(4000), nullable=False),
    Column('time', TIMESTAMP, nullable=False, default='now()'),
    Column('level', Integer, nullable=False),
    Column('component', Integer, nullable=False),
    Column('module', String(200), nullable=False),
    Column('line', Integer, nullable=False),
    Column('task', Integer),
    Column('computer', String(100))
)

project = Table(
    'project', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(180), nullable=False),
    Column('class_names', String(8000), nullable=False),
    Column('ignore_folders', String(8000), nullable=False)
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
    Column('project', Integer, nullable=False),
    Column('dag', Integer, nullable=False),
    Column('part', String(30)),
    Column('y_pred', Integer),
    Column('y', Integer),
    Column('metric_diff', Float),
    Column('attr1', Float),
    Column('attr2', Float),
    Column('attr3', Float),
    Column('size', BigInteger, nullable=False),
)

report_layout = Table(
    'report_layout', meta,
    Column('name', String(400), primary_key=True),
    Column('content', String(8000), nullable=False),
    Column('last_modified', TIMESTAMP, nullable=False, default='now()')
)

report_series = Table(
    'report_series', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('task', Integer, nullable=False),
    Column('time', TIMESTAMP, nullable=False, default='now()'),
    Column('epoch', Integer, nullable=False),
    Column('value', Float, nullable=False),
    Column('name', String(100), nullable=False),
    Column('part', String(50), nullable=False),
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
    Column('pid', Integer),
    Column('worker_index', Integer),
    Column('additional_info', String(16000)),
    Column('docker_assigned', String(100)),
    Column('type', Integer, nullable=False),
    Column('score', Float),
    Column('report', Integer),
    Column('gpu_assigned', Integer),
    Column('parent', Integer)
)

task_dependency = Table(
    'task_dependency', meta,
    Column('task_id', Integer, primary_key=True),
    Column('depend_id', Integer, primary_key=True),
)

docker = Table(
    'docker', meta,
    Column('name', String(100), primary_key=True),
    Column('computer', String(100), primary_key=True),
    Column('last_activity', TIMESTAMP, nullable=False),
)

model = Table(
    'model', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String(500), nullable=False),
    Column('score_local', Float, nullable=False),
    Column('score_public', Float),
    Column('dag', Integer, nullable=False),
    Column('project', Integer, nullable=False),
    Column('created', TIMESTAMP, nullable=False),
    Column('interface', String(100), nullable=False),
    Column('pred_file_valid', String(200)),
    Column('pred_file_test', String(200)),
    Column('interface_params', String(4000)),
    Column('slot', String(100), nullable=False)
)


def upgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()

    try:
        meta.bind = conn

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
        report_layout.create()
        docker.create()
        model.create()
        auxiliary.create()

        ForeignKeyConstraint([computer_usage.c.computer],
                             [computer.c.name],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([computer.c.syncing_computer],
                             [computer.c.name],
                             ondelete='CASCADE').create()

        Index('computer_name_idx', computer.c.name).create()

        Index('computer_usage_time_idx', computer_usage.c.time.desc()).create()
        Index('computer_usage_id_idx', computer_usage.c.id.desc()).create()

        ForeignKeyConstraint([dag.c.project], [project.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([dag.c.report], [report.c.id],
                             ondelete='CASCADE').create()

        Index('dag_project_idx', dag.c.project.desc()).create()
        Index('dag_created_idx', dag.c.created.desc()).create()
        ForeignKeyConstraint([dag_library.c.dag], [dag.c.id],
                             ondelete='CASCADE').create()
        Index('dag_library_task_idx', dag_library.c.dag.desc()).create()
        Index('dag_id_idx', dag.c.id.desc()).create()

        ForeignKeyConstraint([dag_storage.c.dag], [dag.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([dag_storage.c.file], [file.c.id],
                             ondelete='CASCADE').create()
        Index('dag_storage_dag_idx', dag_storage.c.dag.desc()).create()
        Index('dag_storage_id_idx', dag_storage.c.id.desc()).create()

        ForeignKeyConstraint([file.c.project], [project.c.id],
                             ondelete='CASCADE').create()
        Index('file_created_idx', file.c.created.desc()).create()
        Index('file_project_idx', file.c.project.desc()).create()
        UniqueConstraint(file.c.md5, file.c.project,
                         name='file_md5_idx').create()
        Index('file_id_idx', file.c.id.desc()).create()

        ForeignKeyConstraint([log.c.task], [task.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([log.c.step], [step.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([log.c.computer], [computer.c.name],
                             ondelete='CASCADE').create()

        Index('log_step_idx', log.c.step.desc()).create()
        Index('log_time_idx', log.c.time.desc()).create()
        Index('log_id_idx', log.c.id.desc()).create()

        Index('project_id_idx', project.c.id.desc()).create()
        UniqueConstraint(project.c.name, name='project_name').create()

        ForeignKeyConstraint([report.c.project], [project.c.id],
                             ondelete='CASCADE').create()
        Index('report_id_idx', report.c.id.desc()).create()

        ForeignKeyConstraint([report_img.c.project], [project.c.id],
                             ondelete='CASCADE').create()
        Index('report_img_project_idx', report_img.c.project.desc()).create()
        Index('report_img_task_idx', report_img.c.task.desc()).create()
        Index('report_img_id_idx', report_img.c.id.desc()).create()

        ForeignKeyConstraint([report_series.c.task], [task.c.id],
                             ondelete='CASCADE').create()
        Index('report_series_id_idx', report_series.c.id.desc()).create()

        ForeignKeyConstraint([report_task.c.task], [task.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([report_task.c.report], [report.c.id],
                             ondelete='CASCADE').create()
        Index('report_task_id_idx', report_task.c.id.desc()).create()

        ForeignKeyConstraint([step.c.task], [task.c.id],
                             ondelete='CASCADE').create()
        Index('step_name_idx', step.c.name).create()
        Index('step_id_idx', step.c.id.desc()).create()

        ForeignKeyConstraint([task.c.computer], [computer.c.name],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([task.c.report], [report.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([task.c.computer_assigned], [computer.c.name],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([task.c.dag], [dag.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([task.c.parent], [task.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint(
            [task.c.docker_assigned, task.c.computer_assigned],
            [docker.c.name, docker.c.computer], ondelete='CASCADE').create()

        Index('task_status_idx', task.c.status).create()
        Index('task_dag_idx', task.c.dag.desc()).create()
        Index('task_finished_idx', task.c.finished.desc()).create()
        Index('task_name_idx', task.c.name).create()
        Index('task_started_idx', task.c.started.desc()).create()
        Index('task_id_idx', task.c.id.desc()).create()
        Index('task_docker_idx', task.c.docker_assigned,
              task.c.computer_assigned).create()

        Index('task_dependency_task_idx',
              task_dependency.c.task_id.desc()).create()
        Index('task_dependency_depend_idx',
              task_dependency.c.depend_id.desc()).create()

        ForeignKeyConstraint([task_dependency.c.task_id], [task.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([task_dependency.c.depend_id], [task.c.id],
                             ondelete='CASCADE').create()

        ForeignKeyConstraint([docker.c.computer], [computer.c.name],
                             ondelete='CASCADE').create()

        ForeignKeyConstraint([model.c.dag], [dag.c.id],
                             ondelete='CASCADE').create()
        ForeignKeyConstraint([model.c.project], [project.c.id],
                             ondelete='CASCADE').create()
        UniqueConstraint(model.c.project, model.c.name,
                         name='model_project_name_unique').create()
    except:
        trans.rollback()
        raise
    else:
        trans.commit()


def downgrade(migrate_engine):
    conn = migrate_engine.connect()
    trans = conn.begin()
    meta.bind = conn
    try:
        auxiliary.drop()
        model.drop()
        docker.drop()
        computer_usage.drop()
        dag_library.drop()
        dag_storage.drop()
        log.drop()
        file.drop()
        report_task.drop()
        report_img.drop()
        report_series.drop()
        report_layout.drop()
        report.drop()
        task_dependency.drop()
        step.drop()
        task.drop()
        dag.drop()
        computer.drop()
        project.drop()
    except:
        trans.rollback()
        raise
    else:
        trans.commit()
