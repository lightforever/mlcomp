from mlcomp.utils.misc import now
from mlcomp.db.models.base import *
import sys


class ReportSeries(Base):
    __tablename__ = 'report_series'

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    value = sa.Column(sa.Float)
    epoch = sa.Column(sa.Integer)
    time = sa.Column(sa.DateTime)
    task = sa.Column(sa.Integer, ForeignKey('task.id'))
    part = sa.Column(sa.String)

    task_rel = relationship('Task', lazy='noload')


class ReportImg(Base):
    __tablename__ = 'report_img'

    id = sa.Column(sa.Integer, primary_key=True)
    group = sa.Column(sa.String)
    epoch = sa.Column(sa.Integer)
    task = sa.Column(sa.Integer, ForeignKey('task.id'))
    img = sa.Column(sa.LargeBinary)
    dag = sa.Column(sa.Integer, ForeignKey('dag.id'))
    part = sa.Column(sa.String)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))
    y_pred = sa.Column(sa.Integer)
    y = sa.Column(sa.Integer)
    metric_diff = sa.Column(sa.Float)
    attr1 = sa.Column(sa.Float)
    attr2 = sa.Column(sa.Float)
    attr3 = sa.Column(sa.Float)
    size = sa.Column(sa.BigInteger)

    def __init__(self, **kwargs):
        super(ReportImg, self).__init__(**kwargs)
        self.size = sys.getsizeof(self.img)


class Report(Base):
    __tablename__ = 'report'

    id = sa.Column(sa.Integer, primary_key=True)
    config = sa.Column(sa.String)
    time = sa.Column(sa.DateTime, default=now())
    name = sa.Column(sa.String)
    project = sa.Column(sa.Integer, ForeignKey('project.id'))


class ReportTasks(Base):
    __tablename__ = 'report_task'

    id = sa.Column(sa.Integer, primary_key=True)
    report = sa.Column(sa.Integer, ForeignKey('report.id'))
    task = sa.Column(sa.Integer, ForeignKey('task.id'))


class ReportScheme(Base):
    __tablename__ = 'report_scheme'

    name = sa.Column(sa.String, primary_key=True)
    content = sa.Column(sa.LargeBinary)
    last_modified = sa.Column(sa.TIMESTAMP)


__all__ = ['ReportSeries',
           'ReportImg',
           'ReportTasks',
           'Report',
           'ReportScheme']
