from typing import List
from itertools import groupby

from mlcomp.db.models import ReportSeries, Task
from mlcomp.db.providers.base import BaseDataProvider


class ReportSeriesProvider(BaseDataProvider):
    model = ReportSeries

    def by_dag(self, dag: int, metrics: List[str]):
        tasks = self.query(Task.id, Task.name).filter(Task.dag == dag).all()
        ids = [id for id, _ in tasks]

        series = self.query(ReportSeries).filter(
            ReportSeries.task.in_(ids)).filter(
            ReportSeries.name.in_(metrics)).order_by(ReportSeries.task).all()

        res = []
        for task_id, task_series in groupby(series, key=lambda x: x.task):
            task_series = list(task_series)
            task_series = sorted(task_series, key=lambda x: x.name)

            for name, name_series in groupby(task_series,
                                             key=lambda x: x.name):
                name_series = list(name_series)
                name_series = sorted(name_series, key=lambda x: x.part)
                groups = []
                for group, part_series in groupby(name_series,
                                                  key=lambda x: x.part):
                    part_series = list(part_series)
                    part_series = sorted(part_series, key=lambda x: x.epoch)
                    group = {'name': group,
                             'epoch': [s.epoch for s in part_series],
                             'value': [s.value for s in part_series]
                             }
                    groups.append(group)
                res.append((task_id, name, groups))
        return res


__all__ = ['ReportSeriesProvider']
