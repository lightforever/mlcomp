import base64
import pickle
from itertools import groupby
from typing import List

from sqlalchemy import func, case
from sqlalchemy.orm import joinedload

from mlcomp.db.core import PaginatorOptions, Session
from mlcomp.db.enums import TaskStatus, TaskType
from mlcomp.db.models import Report, ReportTasks, Task, ReportSeries, \
    ReportImg, ReportLayout
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.db.report_info import ReportLayoutSeries, ReportLayoutInfo
from mlcomp.db.report_info.item import ReportLayoutItem
from mlcomp.utils.io import yaml_load, yaml_dump


class ReportProvider(BaseDataProvider):
    model = Report

    def __init__(self, session: Session = None):
        super(ReportProvider, self).__init__(session)

    def get(self, filter: dict, options: PaginatorOptions):
        task_count_cond = func.sum(
            case(
                whens=[(Task.status <= TaskStatus.InProgress.value, 1)],
                else_=0
            ).label('tasks_not_finished')
        )

        query = self.query(Report,
                           func.count(ReportTasks.task).label('tasks_count'),
                           task_count_cond, ). \
            join(ReportTasks, ReportTasks.report == Report.id, isouter=True). \
            join(Task, Task.id == ReportTasks.task, isouter=True)

        if filter.get('task'):
            query = query.filter(ReportTasks.task == filter['task'])

        query = query.group_by(Report.id)

        total = query.count()
        data = []
        for report, task_count, tasks_not_finished in self.paginator(
            query, options
        ):
            item = {
                'id': report.id,
                'time': self.serialize_datetime(report.time),
                'tasks': task_count,
                'tasks_not_finished': tasks_not_finished,
                'name': report.name
            }
            data.append(item)

        return {'total': total, 'data': data}

    def _detail_series(
        self, series: List[ReportSeries], r: ReportLayoutSeries
    ):
        series = [s for s in series if s.name == r.name]
        res = []

        series = sorted(series, key=lambda x: x.part)
        series_group = groupby(series, key=lambda x: x.part)
        for key, group in series_group:
            group = list(group)
            group = sorted(group, key=lambda x: x.task)
            for task_key, group_task in groupby(group, key=lambda x: x.task):
                group_task = list(group_task)
                res.append(
                    {
                        'x': [item.epoch for item in group_task],
                        'y': [item.value for item in group_task],
                        'stage': [item.stage for item in group_task],
                        'color': 'orange' if key == 'valid' else 'blue',
                        'time': [
                            self.serialize_datetime(item.time)
                            for item in group_task
                        ],
                        'group': key,
                        'task_name': group_task[0].task_rel.name,
                        'task_id': task_key,
                        'source': r.key,
                        'name': r.name
                    }
                )

        return res

    def _detail_single_img(self, report: int, item: ReportLayoutItem):
        res = []
        img_objs = self.query(ReportImg). \
            filter(ReportImg.group == item.name).all()

        for img_obj in img_objs:
            img_decoded = pickle.loads(img_obj.img)
            item = {
                'name': f'{img_obj.group} - {img_obj.part}',
                'data': base64.b64encode(img_decoded['img']).decode('utf-8')
            }
            res.append(item)

        return res

    def detail_img_classify_descr(self, report: int, item: ReportLayoutItem):
        res = []
        tasks = self.query(ReportTasks.task.distinct()
                           ).filter(ReportTasks.report == report).all()
        tasks = [t[0] for t in tasks]
        task_names = {
            id: name
            for id, name in self.query(Task.id, Task.name
                                       ).filter(Task.id.in_(tasks)).all()
        }
        for task in tasks:
            obj = {
                'name': task_names[task],
                'group': item.name,
                'task': task,
            }
            res.append(obj)
        return res

    def detail(self, id: int):
        report_obj = self.by_id(id)
        tasks = self.query(ReportTasks.task).filter(ReportTasks.report == id
                                                    ).all()
        tasks = [t[0] for t in tasks]
        config = yaml_load(report_obj.config)
        report = ReportLayoutInfo(config)

        series = self.query(ReportSeries). \
            filter(ReportSeries.task.in_(tasks)). \
            order_by(ReportSeries.epoch). \
            options(joinedload(ReportSeries.task_rel, innerjoin=True)).all()

        items = dict()
        for s in report.series:
            items[s.name] = self._detail_series(series, s)

        for element in report.precision_recall + report.f1:
            items[element.name] = self._detail_single_img(id, element)

        for element in report.img_classify:
            items[element.name] = self.detail_img_classify_descr(id, element)

        return {
            'data': items,
            'layout': report.layout,
            'metric': report.metric.serialize()
        }

    def add_dag(self, dag: int, report: int):
        tasks = self.query(Task.id). \
            filter(Task.dag == dag). \
            filter(Task.type <= TaskType.Train.value). \
            all()

        report_tasks = self.query(ReportTasks.task
                                  ).filter(ReportTasks.report == report).all()

        for t in set(t[0] for t in tasks) - set(t[0] for t in report_tasks):
            self.add(ReportTasks(report=report, task=t))

    def remove_dag(self, dag: int, report: int):
        tasks = self.query(Task.id).filter(Task.dag == dag).all()
        tasks = [t[0] for t in tasks]
        self.query(ReportTasks).filter(ReportTasks.report == report). \
            filter(ReportTasks.task.in_(tasks)). \
            delete(synchronize_session=False)

        self.session.commit()

    def remove_task(self, task: int, report: int):
        self.query(ReportTasks).filter(ReportTasks.report == report). \
            filter(ReportTasks.task == task). \
            delete(synchronize_session=False)

        self.session.commit()

    def add_task(self, task: int, report: int):
        self.add(ReportTasks(task=task, report=report))
        self.session.commit()

    def update_layout_start(self, id: int):
        layouts = self.query(ReportLayout.name).all()
        report = self.by_id(id)
        layouts = [l[0] for l in layouts]
        if report.layout in layouts:
            layouts.remove(report.layout)
            layouts.insert(0, report.layout)

        return {'id': id, 'layouts': layouts}

    def update_layout_end(self, id: int, layout: str, layouts: dict):
        layout_content = yaml_dump(layouts[layout])
        report = self.by_id(id)
        report.config = layout_content
        report.layout = layout
        self.commit()


__all__ = ['ReportProvider']
