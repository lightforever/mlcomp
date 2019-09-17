import pickle
from typing import Tuple

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

from mlcomp.db.core import Session
from mlcomp.db.models import Report, ReportImg, Task, ReportTasks, ReportSeries
from mlcomp.db.providers import ReportProvider, \
    ReportLayoutProvider, \
    TaskProvider,\
    ReportImgProvider, \
    ReportTasksProvider, \
    ReportSeriesProvider, \
    DagProvider
from mlcomp.utils.img import resize_saving_ratio
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.misc import now


class ClassificationReportBuilder:
    def __init__(
        self,
        session: Session,
        task: Task,
        layout: str,
        part: str = 'valid',
        name: str = 'img_classify',
        max_img_size: Tuple[int, int] = None,
        main_metric: str = 'accuracy',
        plot_count: int = 0
    ):
        self.session = session
        self.task = task
        self.layout = layout
        self.part = part
        self.name = name or 'img_classify'
        self.max_img_size = max_img_size
        self.main_metric = main_metric
        self.plot_count = plot_count

        self.dag_provider = DagProvider(session)
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
        self.task_provider.update()

    def process_scores(self, scores):
        for key, item in self.layout_dict['items'].items():
            item['name'] = key
            if item['type'] == 'series' and item['key'] in scores:
                series = ReportSeries(
                    name=item['name'],
                    value=float(scores[item['key']]),
                    epoch=0,
                    time=now(),
                    task=self.task.id,
                    part='valid',
                    stage='stage1'
                )

                self.report_series_provider.add(series)

    def process_pred(self, imgs: np.array, preds: np.array,
                     targets: np.array = None, attrs=None, scores=None):
        for key, item in self.layout_dict['items'].items():
            item['name'] = key
            if item['type'] != 'img_classify':
                continue

            report_imgs = []
            dag = self.dag_provider.by_id(self.task.dag)

            for i in range(len(imgs)):
                if self.plot_count <= 0:
                    break

                img = resize_saving_ratio(imgs[i], self.max_img_size)
                pred = preds[i]
                attr = attrs[i] if attrs else {}

                y = None
                score = None
                if targets is not None:
                    y = targets[i]
                    score = float(scores[self.main_metric][i])

                y_pred = pred.argmax()
                retval, buffer = cv2.imencode('.jpg', img)
                report_img = ReportImg(
                    group=item['name'],
                    epoch=0,
                    task=self.task.id,
                    img=buffer,
                    dag=self.task.dag,
                    part=self.part,
                    project=self.project,
                    y_pred=y_pred,
                    y=y,
                    score=score,
                    **attr
                )

                report_imgs.append(report_img)
                dag.img_size += report_img.size

            self.dag_provider.commit()
            self.report_img_provider.bulk_save_objects(report_imgs)

            if targets is not None and item.get('confusion_matrix'):
                matrix = confusion_matrix(
                    targets,
                    preds.argmax(axis=1),
                    labels=np.arange(preds.shape[1])
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

            self.plot_count -= 1


__all__ = ['ClassificationReportBuilder']
