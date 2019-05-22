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
                'time': report.time,
                'tasks': task_count,
                'tasks_not_finished': tasks_not_finished
            }
            data.append(item)

        return {'total': total, 'data': data}

    def detail(self, id: int):
        report = self.by_id(id)
        tasks = self.query(ReportTasks.task).filter(ReportTasks.report == id).all()
        config = json.loads(report.config)
        res = {
            'rows': []
        }

        for row in config.values():
            r = {'items': []}
            for k, v in row.items():
                item = {'name': k, 'type': v['type']}
                if item['type']=='series':
                    series = self.query(ReportSeries).filter(ReportSeries.task.in_(tasks)).\
                        order_by(ReportSeries.epoch).all()
                    data = []
                    series_task_group = groupby(series, key=lambda x: (x.task, x.group))
                    for group_key, group in series_task_group:
                        data.append(
                            {
                                'x': [item.epoch for item in group],
                                'y': [item.value for item in group],
                                'color': 'orange' if group_key[1]=='valid' else 'blue'
                            })

                    item['data'] = data

                r['items'].append(item)

            res['rows'].append(r)

        return res


class ReportTasksProvider(BaseDataProvider):
    model = ReportTasks
