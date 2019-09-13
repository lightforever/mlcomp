import pickle
from collections import OrderedDict
from typing import Tuple, List

import cv2
import numpy as np

from mlcomp.db.core import Session
from mlcomp.db.models import Report, ReportImg, Task, ReportTasks, ReportSeries
from mlcomp.db.providers import ReportProvider, ReportLayoutProvider, \
    TaskProvider, ReportImgProvider, ReportTasksProvider, \
    ReportSeriesProvider, DagProvider
from mlcomp.utils.img import resize_saving_ratio
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.misc import now


class SegmentationReportBuilder:
    def __init__(
            self,
            session: Session,
            task: Task,
            layout: str,
            part: str = 'valid',
            name: str = 'img_segment',
            max_img_size: Tuple[int, int] = None,
            stack_type: str = 'vertical',
            main_metric: str = 'dice',
            plot_count: int = 0,
            colors: List[Tuple] = None
    ):
        self.session = session
        self.task = task
        self.layout = layout
        self.part = part
        self.name = name
        self.max_img_size = max_img_size
        self.stack_type = stack_type
        self.main_metric = main_metric
        self.colors = colors
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

        self.create_base()

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

    def encode_pred(self, mask: np.array):
        res = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
        for i, c in enumerate(mask):
            c = np.repeat(c[:, :, None], 3, axis=2)
            color = self.colors[i] if self.colors is not None else (
                255, 255, 255
            )
            res += (c * color).astype(np.uint8)

        return res

    def plot_mask(self, img: np.array, mask: np.array):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

        for i, c in enumerate(mask):
            contours, _ = cv2.findContours(
                c, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
            color = self.colors[i] if self.colors else (0, 255, 0)
            for i in range(0, len(contours)):
                cv2.polylines(img, contours[i], True, color, 2)

        return img

    def process_scores(self, scores):
        for key, item in self.layout_dict['items'].items():
            item['name'] = key
            if item['type'] == 'series' and item['key'] in scores:
                series = ReportSeries(
                    name=item['name'],
                    value=scores[item['key']],
                    epoch=0,
                    time=now(),
                    task=self.task.id,
                    part='valid',
                    stage='stage1'
                )

                self.report_series_provider.add(series)

    def process_pred(self, imgs: np.array, preds: dict,
                     targets: np.array = None, attrs=None, scores=None):
        for key, item in self.layout_dict['items'].items():
            item['name'] = key
            if item['type'] != 'img_segment':
                continue

            report_imgs = []
            dag = self.dag_provider.by_id(self.task.dag)

            for i in range(len(imgs)):
                if self.plot_count <= 0:
                    break

                if targets is not None:
                    img = self.plot_mask(imgs[i], targets[i])
                else:
                    img = imgs[i]

                imgs_add = [img]
                for key, value in preds.items():
                    imgs_add.append(self.encode_pred(value[i]))

                for j in range(len(imgs_add)):
                    imgs_add[j] = resize_saving_ratio(imgs_add[j],
                                                      self.max_img_size)

                if self.stack_type == 'horizontal':
                    img = np.hstack(imgs_add)
                else:
                    img = np.vstack(imgs_add)

                attr = attrs[i] if attrs else {}

                score = None
                if targets is not None:
                    score = scores[self.main_metric][i]

                retval, buffer = cv2.imencode('.jpg', img)
                report_img = ReportImg(
                    group=item['name'],
                    epoch=0,
                    task=self.task.id,
                    img=buffer,
                    dag=self.task.dag,
                    part=self.part,
                    project=self.project,
                    score=score,
                    **attr
                )

                self.plot_count -= 1
                report_imgs.append(report_img)
                dag.img_size += report_img.size

            self.dag_provider.commit()
            self.report_img_provider.bulk_save_objects(report_imgs)


__all__ = ['SegmentationReportBuilder']
