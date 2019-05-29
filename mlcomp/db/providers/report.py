from mlcomp.db.providers.base import *
import json
from itertools import groupby
from typing import List
import pickle
from mlcomp.db.misc.report_info import ReportInfo, ReportInfoSeries, ReportInfoItem
import base64


class ReportSeriesProvider(BaseDataProvider):
    model = ReportSeries


class ReportImgProvider(BaseDataProvider):
    model = ReportImg

    def add_or_replace(self, obj: ReportImg):
        query = self.query(ReportImg).filter(ReportImg.task == obj.task).filter(
            ReportImg.group == obj.group).filter(ReportImg.number == obj.number)
        if query.count() == 0:
            self.add(obj)
        else:
            query.update({'img': obj.img, 'epoch': obj.epoch})
            self.session.commit()

    def remove(self, filter: dict):
        query = self.query(ReportImg)
        if filter.get('dag'):
            query = query.filter(ReportImg.dag == filter['dag'])
        if filter.get('project'):
            query = query.filter(ReportImg.project == filter['project'])
        query.delete(synchronize_session=False)
        self.session.commit()


class ReportProvider(BaseDataProvider):
    model = Report

    def __init__(self, session: Session = None):
        super(ReportProvider, self).__init__(session)

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Report, func.count(ReportTasks.task).label('tasks_count'),
                           func.count(ReportTasks.task).filter(Task.status <= TaskStatus.InProgress.value).label(
                               'tasks_not_finished'), ). \
            join(ReportTasks, ReportTasks.report == Report.id, isouter=True). \
            join(Task, Task.id == ReportTasks.task, isouter=True)

        if filter.get('task'):
            query = query.filter(ReportTasks.task == filter['task'])

        query = query.group_by(Report.id)

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

    def _detail_series(self, series: List[ReportSeries], r: ReportInfoSeries):
        item = {'name': r.name, 'type': 'series', 'rows': 1, 'cols': 1}

        series = [s for s in series if s.name == r.name]
        res = []
        if len(set(s.task for s in series)) == 1:
            series = sorted(series, key=lambda x: x.group)
            series_group = groupby(series, key=lambda x: x.group)
            data = []
            for key, group in series_group:
                group = list(group)
                data.append(
                    {
                        'x': [item.epoch for item in group],
                        'y': [item.value for item in group],
                        'color': 'orange' if key == 'valid' else 'blue',
                        'name': key
                    })

            item['data'] = data
            res.append(item)

        else:
            if r.multi == 'none':
                return res

            if r.multi == 'single':
                series = sorted(series, key=lambda x: x.group)
                series_group = groupby(series, key=lambda x: x.group)
                for key, group in series_group:
                    if r.single_group and key != r.single_group:
                        continue
                    data = []
                    group = list(group)
                    item_copy = item.copy()
                    group = sorted(group, key=lambda x: x.task)
                    for task_key, group_task in groupby(group, key=lambda x: x.task):
                        group_task = list(group_task)
                        data.append(
                            {
                                'x': [item.epoch for item in group_task],
                                'y': [item.value for item in group_task],
                                'color': 'orange' if key == 'valid' else 'blue',
                                'name': f'{group_task[0].task_rel.name}'
                            })

                    item_copy['data'] = data
                    item_copy['name'] = f'{r.name},{key}'
                    res.append(item_copy)
            else:
                series = sorted(series, key=lambda x: x.task)
                series_task_group = groupby(series, key=lambda x: x.task)
                for task_key, group_task in series_task_group:
                    data = []
                    item_copy = item.copy()
                    group_task = list(group_task)
                    group_task = sorted(group_task, key=lambda x: x.group)

                    for key, group in groupby(group_task, key=lambda x: x.group):
                        group = list(group)
                        data.append(
                            {
                                'x': [item.epoch for item in group],
                                'y': [item.value for item in group],
                                'color': 'orange' if key == 'valid' else 'blue',
                                'name': key
                            })

                    item_copy['data'] = data
                    item_copy['name'] = f'{r.name},{group_task[0].task_rel.name}'
                    res.append(item_copy)
        return res

    def _detail_single_img(self, task: int, epoch: int, item: ReportInfoItem):
        res = []
        img_objs = self.query(ReportImg).filter(ReportImg.task == task).filter(ReportImg.epoch == epoch). \
            filter(ReportImg.group == item.name).all()
        for img_obj in img_objs:
            img_decoded = pickle.loads(img_obj.img)
            item = {'name': img_obj.group, 'type': 'img', 'rows': 1, 'cols': 1,
                    'data': base64.b64encode(img_decoded['img']).decode('utf-8')}
            res.append(item)

        return res

    def _detail_img_confusion(self, task: int, epoch: int, item: ReportInfoItem):
        res = []
        img_objs = self.query(ReportImg).filter(ReportImg.task == task).filter(ReportImg.epoch == epoch). \
            filter(ReportImg.group == item.name).all()
        imgs = [pickle.loads(img.img) for img in img_objs]
        false_positive = [img for img in imgs if img['y'] == 0 and img['y_pred'] == 1]
        false_negative = [img for img in imgs if img['y'] == 1 and img['y_pred'] == 0]

        for group, name in [(false_positive, 'false_positive'), (false_negative, 'false_negative')]:
            item = {'name': name, 'type': 'img_list', 'rows': 1, 'cols': len(false_positive),
                    'items': [
                        {'img': base64.b64encode(img['img']).decode('utf-8'), 'text': str(img['pred'])[:4]} for img in
                        group
                    ]}
            res.append(item)

        return res

    def detail(self, id: int):
        report_obj = self.by_id(id)
        tasks = self.query(ReportTasks.task).filter(ReportTasks.report == id).all()
        tasks = [t[0] for t in tasks]
        config = json.loads(report_obj.config)
        report = ReportInfo(config)

        series = self.query(ReportSeries).filter(ReportSeries.task.in_(tasks)). \
            order_by(ReportSeries.epoch). \
            options(joinedload(ReportSeries.task_rel)).all()

        best_task_epoch = None
        for s in series:
            if s.name == report.metric.name:
                if best_task_epoch is None or \
                        (best_task_epoch[1] > s.value if report.metric.minimize else best_task_epoch[1] < s.value):
                    best_task_epoch = [(s.task, s.epoch), s.value]

        res = []
        for s in report.series:
            res.extend(self._detail_series(series, s))

        if best_task_epoch:
            for element in report.precision_recall + report.f1:
                res.extend(self._detail_single_img(best_task_epoch[0][0], best_task_epoch[0][1], element))

            for element in report.img_confusion:
                res.extend(self._detail_img_confusion(best_task_epoch[0][0], best_task_epoch[0][1], element))

        return res

    def add_dag(self, dag: int, report: int):
        tasks = self.query(Task.id).filter(Task.dag == dag).all()
        report_tasks = self.query(ReportTasks.task).filter(ReportTasks.report == report).all()
        for t in set(t[0] for t in tasks) - set(t[0] for t in report_tasks):
            self.add(ReportTasks(report=report, task=t))

    def remove_dag(self, dag: int, report: int):
        tasks = self.query(Task.id).filter(Task.dag == dag).all()
        tasks = [t[0] for t in tasks]
        self.query(ReportTasks).filter(ReportTasks.report == report). \
            filter(ReportTasks.task.in_(tasks)).delete(synchronize_session=False)
        self.session.commit()

    def remove_task(self, task: int, report: int):
        self.query(ReportTasks).filter(ReportTasks.report == report). \
            filter(ReportTasks.task == task).delete(synchronize_session=False)
        self.session.commit()

    def add_task(self, task: int, report: int):
        self.add(ReportTasks(task=task, report=report))
        self.session.commit()


class ReportTasksProvider(BaseDataProvider):
    model = ReportTasks
