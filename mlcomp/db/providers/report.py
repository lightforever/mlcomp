from mlcomp.db.providers.base import *
import json
from itertools import groupby
from typing import List
import pickle
from mlcomp.db.misc.report_info import ReportInfo, ReportInfoSeries, ReportInfoItem
import base64
from sqlalchemy import and_


class ReportSeriesProvider(BaseDataProvider):
    model = ReportSeries


class ReportImgProvider(BaseDataProvider):
    model = ReportImg

    def remove(self, filter: dict):
        query = self.query(ReportImg)
        if filter.get('dag'):
            query = query.filter(ReportImg.dag == filter['dag'])
        if filter.get('project'):
            query = query.filter(ReportImg.project == filter['project'])
        query.delete(synchronize_session=False)
        self.session.commit()

    def remove_lower(self, task_id: int, name: str, epoch: int):
        self.query(ReportImg).filter(ReportImg.task == task_id). \
            filter(ReportImg.group == name). \
            filter(ReportImg.epoch < epoch). \
            delete(synchronize_session=False)
        self.session.commit()

    def detail_img_classify(self, filter: dict, options: PaginatorOptions = None):
        res = {'data': []}
        confusion = self.query(ReportImg.img). \
            filter(ReportImg.task == filter['task']). \
            filter(ReportImg.part == filter['part']). \
            filter(ReportImg.group == filter['group'] + '_confusion'). \
            filter(ReportImg.epoch == filter['epoch']).first()
        if confusion:
            confusion = pickle.loads(confusion[0])['data'].tolist()
            res['confusion'] = {'data': confusion}

        res.update(filter)

        query = self.query(ReportImg).filter(ReportImg.task == filter['task']).filter(ReportImg.epoch == filter['epoch']). \
            filter(ReportImg.group == filter['group']).filter(ReportImg.part == filter['part'])

        if filter.get('y') is not None and filter.get('y_pred') is not None:
            query = query.filter(
                and_(ReportImg.y == filter['y'], ReportImg.y_pred == filter['y_pred'])
            )

        if filter.get('metric_diff_min') is not None:
            query = query.filter(ReportImg.metric_diff >= filter['metric_diff_min'])
        if filter.get('metric_diff_max') is not None:
            query = query.filter(ReportImg.metric_diff <= filter['metric_diff_max'])

        project = self.query(Project).join(Dag).join(Task).filter(Task.id == filter['task']).first()
        class_names = pickle.loads(project.class_names)

        res['total'] = query.count()
        res['class_names'] = class_names['default']
        query = self.paginator(query, options)
        img_objs = query.all()
        for img_obj in img_objs:
            img = pickle.loads(img_obj.img)
            res['data'].append({
                'content': base64.b64encode(img['img']).decode('utf-8'),
                'id': img_obj.id,
                'y_pred': img_obj.y_pred,
                'y': img_obj.y,
                'metric_diff': round(img_obj.metric_diff, 2)
            })

        return res


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
                'time': self.serialize_datetime(report.time),
                'tasks': task_count,
                'tasks_not_finished': tasks_not_finished,
                'name': report.name
            }
            data.append(item)

        return {'total': total, 'data': data}

    def _detail_series(self, series: List[ReportSeries], r: ReportInfoSeries):
        series = [s for s in series if s.name == r.name]
        res = []

        series = sorted(series, key=lambda x: x.group)
        series_group = groupby(series, key=lambda x: x.group)
        for key, group in series_group:
            group = list(group)
            group = sorted(group, key=lambda x: x.task)
            for task_key, group_task in groupby(group, key=lambda x: x.task):
                group_task = list(group_task)
                res.append(
                    {
                        'data': {
                            'x': [item.epoch for item in group_task],
                            'y': [item.value for item in group_task],
                            'color': 'orange' if key == 'valid' else 'blue',
                            'time': [self.serialize_datetime(item.time) for item in group_task]
                        },
                        'group': key,
                        'task_name': group_task[0].task_rel.name,
                        'task_id': task_key,
                        'source': r.name
                    }
                )

        return res

    def _detail_single_img(self, task: int, epoch: int, item: ReportInfoItem):
        res = []
        img_objs = self.query(ReportImg).filter(ReportImg.task == task).filter(ReportImg.epoch == epoch). \
            filter(ReportImg.group == item.name).all()
        for img_obj in img_objs:
            img_decoded = pickle.loads(img_obj.img)
            item = {'name': img_obj.group,
                    'data': base64.b64encode(img_decoded['img']).decode('utf-8')}
            res.append(item)

        return res

    def detail_img_classify_descr(self, task: int, name: str):
        res = []
        parts = self.query(ReportImg.part.distinct()). \
            filter(ReportImg.task == task). \
            filter(ReportImg.group == name). \
            order_by(ReportImg.part).all()
        for part in parts:
            part = part[0]
            res.append({
                'name': part,
                'type': 'img_classify',
                'source': name,
                'data': {
                    'epochs': [e[0] for e in self.query(ReportImg.epoch.distinct()). \
                        filter(ReportImg.task == task). \
                        filter(ReportImg.group == name). \
                        filter(ReportImg.part == part). \
                        order_by(ReportImg.epoch).all()],
                    'task': task,
                    'group': name,
                    'part': part
                }

            })
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

        items = dict()
        for s in report.series:
            items[s.name] = self._detail_series(series, s)

        if best_task_epoch:
            for element in report.precision_recall + report.f1:
                items[element.name] = self._detail_single_img(best_task_epoch[0][0], best_task_epoch[0][1], element)

            for element in report.img_classify:
                items[element.name] = self.detail_img_classify_descr(best_task_epoch[0][0], element.name)

        return {'data': items, 'layout': report.layout}

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
