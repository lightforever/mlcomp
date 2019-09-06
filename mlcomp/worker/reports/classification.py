import pickle
from typing import Tuple, List

import numpy as np
from sklearn.metrics import confusion_matrix

from mlcomp.db.core import Session
from mlcomp.db.models import Report, ReportImg, Task, ReportTasks, ReportSeries
from mlcomp.db.providers import ReportProvider, ReportLayoutProvider, \
    TaskProvider, ReportImgProvider, ReportTasksProvider, ReportSeriesProvider
from mlcomp.utils.img import resize_saving_ratio
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.misc import now


class ClassificationReportBuilder:
    def __init__(
        self,
        session: Session,
        task: Task,
        layout: str,
        imgs: np.array,
        preds: np.array,
        targets: np.array = None,
        part: str = 'valid',
        name: str = 'img_classify',
        scores: dict = None,
        max_img_size: Tuple[int, int] = None,
        attrs: List[dict] = None
    ):
        self.session = session
        self.task = task
        self.layout = layout
        self.imgs = imgs
        self.preds = preds
        self.targets = targets
        self.part = part
        self.name = name
        self.scores = scores
        self.max_img_size = max_img_size
        self.attrs = attrs

        self.report_provider = ReportProvider(session)
        self.layout_provider = ReportLayoutProvider(session)
        self.task_provider = TaskProvider(session)
        self.report_img_provider = ReportImgProvider(session)
        self.report_task_provider = ReportTasksProvider(session)
        self.report_series_provider = ReportSeriesProvider(session)

        self.project = self.task_provider.project(task.id).id
        self.layout = self.layout_provider.by_name(layout)
        self.layout_dict = yaml_load(self.layout.content)

    def create_base(self):
        report = Report(
            config=yaml_dump(self.layout_dict),
            time=now(),
            layout=self.layout.name,
            project=self.project,
            name=self.name
        )
        self.report_provider.add(report)
        self.report_task_provider.add(
            ReportTasks(report=report.id, task=self.task.id)
        )

        self.task.report = report.id
        self.task.name = self.name
        self.task_provider.update()

    def process_series(self, item: dict):
        series = ReportSeries(
            name=item['name'],
            value=self.scores.get(item['key'], 0),
            epoch=0,
            time=now(),
            task=self.task.id,
            part='valid',
            stage='stage1'
        )

        self.report_series_provider.add(series)

    def process_img_classify(self, item: dict):
        for i in range(len(self.imgs)):
            img = resize_saving_ratio(self.imgs[i], self.max_img_size)
            pred = self.preds[i]
            attrs = self.attrs[i] if self.attrs else {}

            y = None
            metric_diff = None
            if self.targets is not None:
                y = self.targets[i]
                metric_diff = 1 - pred[y]

            y_pred = pred.argmax()
            report_img = ReportImg(
                group='img_classify',
                epoch=0,
                task=self.task.id,
                img=pickle.dumps({'img': img}),
                dag=self.task.dag,
                part=self.part,
                project=self.project,
                y_pred=y_pred,
                y=y,
                metric_diff=metric_diff,
                **attrs
            )

            self.report_img_provider.add(report_img)

        if item.get('confusion_matrix'):
            matrix = confusion_matrix(
                self.targets,
                self.preds.argmax(axis=1),
                labels=np.arange(self.preds.shape[1])
            )
            matrix = np.array(matrix)
            c = {'data': matrix}
            obj = ReportImg(
                group=item['name'] + '_confusion',
                epoch=0,
                task=self.task.id,
                img=pickle.dumps(c),
                project=self.project,
                dag=self.task.dag,
                part=self.part
            )
            self.report_img_provider.add(obj)


    def build(self):
        self.create_base()

        for key, item in self.layout_dict['items'].items():
            item['name'] = key
            if item['type'] == 'series':
                self.process_series(item)
            elif item['type'] == 'img_classify':
                self.process_img_classify(item)


__all__ = ['ClassificationReportBuilder']
