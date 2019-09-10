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
        imgs: np.array,
        preds: OrderedDict,
        targets: np.array = None,
        part: str = 'valid',
        name: str = 'img_segment',
        scores: dict = None,
        max_img_size: Tuple[int, int] = None,
        attrs: List[dict] = None,
        stack_type: str = 'vertical',
        main_metric: str = 'dice',
        plot_count: int = 0,
        colors: List[Tuple] = None
    ):
        self.session = session
        self.task = task
        self.layout = layout
        self.imgs = imgs
        self.preds = preds
        self.targets = targets
        self.part = part
        self.name = name
        self.scores = scores or {}
        self.max_img_size = max_img_size
        self.attrs = attrs
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
            value=float(np.mean(self.scores.get(item['key'], [0]))),
            epoch=0,
            time=now(),
            task=self.task.id,
            part='valid',
            stage='stage1'
        )

        self.report_series_provider.add(series)

    def encode_pred(self, mask: np.array):
        res = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
        for i, c in enumerate(mask):
            c = np.repeat(c[:,:,None], 3, axis=2)
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

    def process_img_segment(self, item: dict):
        report_imgs = []
        dag = self.dag_provider.by_id(self.task.dag)

        for i in range(len(self.imgs)):
            if i >= self.plot_count:
                break

            if self.targets is not None:
                img = self.plot_mask(self.imgs[i], self.targets[i])
            else:
                img = self.imgs[i]

            imgs = [img]
            for key, value in self.preds.items():
                imgs.append(self.encode_pred(value[i]))

            for j in range(len(imgs)):
                imgs[j] = resize_saving_ratio(imgs[j], self.max_img_size)

            if self.stack_type == 'horizontal':
                img = np.hstack(imgs)
            else:
                img = np.vstack(imgs)

            attrs = self.attrs[i] if self.attrs else {}

            score = None
            if self.targets is not None:
                score = self.scores[self.main_metric][i]

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
                **attrs
            )

            report_imgs.append(report_img)
            dag.img_size += report_img.size

        self.dag_provider.commit()
        self.report_img_provider.bulk_save_objects(report_imgs)

    def build(self):
        self.create_base()

        for key, item in self.layout_dict['items'].items():
            item['name'] = key
            if item['type'] == 'series':
                self.process_series(item)
            elif item['type'] == 'img_segment':
                self.process_img_segment(item)


__all__ = ['SegmentationReportBuilder']
