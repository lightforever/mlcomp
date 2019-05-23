from mlcomp.db.providers.base import *
import json
from itertools import groupby

class ReportSeriesProvider(BaseDataProvider):
    model = ReportSeries


class ReportImgProvider(BaseDataProvider):
    model = ReportImg


class ReportProvider(BaseDataProvider):
    model = Report

    def __init__(self, session: Session=None):
        super(ReportProvider, self).__init__(session)

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(
            Report,
            func.count(ReportTasks.task).label('tasks_count'),
            func.count(ReportTasks.task).filter(Task.status <= TaskStatus.InProgress.value).label('tasks_not_finished'),
        ).join(ReportTasks, isouter=True).join(Task, isouter=True).group_by(Report.id)

        total = query.count()
        data = []
        for report, task_count, tasks_not_finished in self.paginator(query, options):
            item = {
                'id': report.id,
                'time': self.serializer.serialize_date(report.time),
                'tasks': task_count,
                'tasks_not_finished': tasks_not_finished,
                'name': report.name
            }
            data.append(item)

        return {'total': total, 'data': data}

    def detail(self, id: int):
        report = self.by_id(id)
        tasks = self.query(ReportTasks.task).filter(ReportTasks.report == id).all()
        tasks = [t[0] for t in tasks]
        config = json.loads(report.config)
        res = []

        col_count = 3
        col = 0
        for k, v in config.items():
            item = {'name': k, 'type': v['type'], 'rows': 1, 'cols': 1}
            if item['type']=='series':
                series = self.query(ReportSeries).filter(ReportSeries.task.in_(tasks)).\
                    filter(ReportSeries.name==v['key']).\
                    order_by(ReportSeries.group).all()
                data = []
                series_task_group = groupby(series, key=lambda x: x.group)
                for group_key, group in series_task_group:
                    group = list(group)
                    group = sorted(group, key=lambda x: x.epoch)
                    data.append(
                        {
                            'x': [item.epoch for item in group],
                            'y': [item.value for item in group],
                            'color': 'orange' if group_key=='valid' else 'blue',
                            'name': group_key
                        })

                item['data'] = data

                res.append(item)
                col += 1

        return res

    def add_dag(self, dag: int, report: int):
        tasks = self.query(Task.id).filter(Task.dag==dag).all()
        report_tasks = self.query(ReportTasks.task).filter(ReportTasks.report==report).all()
        for t in set(t[0] for t in tasks) - set(t[0] for t in report_tasks):
            self.add(ReportTasks(report=report, task=t))

    def remove_dag(self,  dag: int, report: int):
        tasks = self.query(Task.id).filter(Task.dag==dag).all()
        tasks = [t[0] for t in tasks]
        self.query(ReportTasks).filter(ReportTasks.report==report).\
            filter(ReportTasks.task.in_(tasks)).delete(synchronize_session=False)
        self.session.commit()

    def remove_task(self, task:int, report:int):
        self.query(ReportTasks).filter(ReportTasks.report == report). \
            filter(ReportTasks.task==task).delete(synchronize_session=False)
        self.session.commit()

    def add_task(self, task:int, report:int):
        self.add(ReportTasks(task=task, report=report))
        self.session.commit()

class ReportTasksProvider(BaseDataProvider):
    model = ReportTasks
