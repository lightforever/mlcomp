import base64
import pickle

import cv2
import numpy as np
from sqlalchemy import and_, or_

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import Project, Dag, ReportImg, Task
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.img import resize_saving_ratio
from mlcomp.utils.io import yaml_load


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

        query = self.query(Dag)
        if filter.get('dag'):
            query.filter(Dag.id == filter['dag']).update({'img_size': 0})

        if filter.get('project'):
            query.filter(Dag.project == filter['project']
                         ).update({'img_size': 0})

        self.session.commit()

    def remove_lower(self, task_id: int, name: str, epoch: int):
        self.query(ReportImg).filter(ReportImg.task == task_id). \
            filter(ReportImg.group == name). \
            filter(ReportImg.epoch < epoch). \
            delete(synchronize_session=False)
        self.session.commit()

    def detail_img_classify(
        self, filter: dict, options: PaginatorOptions = None
    ):
        res = {'data': []}
        confusion = self.query(ReportImg.img). \
            filter(ReportImg.task == filter['task']). \
            filter(ReportImg.group == filter['group'] + '_confusion').first()

        if confusion:
            confusion = pickle.loads(confusion[0])['data']
            res['confusion'] = {'data': confusion.tolist()}

        res.update(filter)

        query = self.query(ReportImg).filter(
            ReportImg.task == filter['task']). \
            filter(ReportImg.group == filter['group'])

        if filter.get('y') is not None and filter.get('y_pred') is not None:
            query = query.filter(
                and_(
                    ReportImg.y == filter['y'],
                    ReportImg.y_pred == filter['y_pred']
                )
            )

        if filter.get('score_min') is not None:
            query = query.filter(
                or_(
                    ReportImg.score >= filter['score_min'],
                    ReportImg.score.__eq__(None)
                )
            )

        if filter.get('score_max') is not None:
            query = query.filter(
                or_(
                    ReportImg.score <= filter['score_max'],
                    ReportImg.score.__eq__(None)
                )
            )

        layout = filter.get('layout')

        if layout and layout.get('attrs'):
            for attr in layout['attrs']:
                field = getattr(ReportImg, attr['source'])
                if attr.get('equal') is not None:
                    query = query.filter(field == attr['equal'])
                if attr.get('greater') is not None:
                    query = query.filter(field >= attr['greater'])
                if attr.get('less') is not None:
                    query = query.filter(field <= attr['less'])

        res['total'] = query.count()

        if confusion is not None:
            project = self.query(Project).join(Dag).join(Task).filter(
                Task.id == filter['task']
            ).first()
            class_names = yaml_load(project.class_names)

            if 'default' in class_names:
                res['class_names'] = class_names['default']
            else:
                res['class_names'] = [
                    str(i) for i in range(confusion.shape[1])
                ]

        query = self.paginator(query, options)
        img_objs = query.all()
        for img_obj in img_objs:
            buffer = img_obj.img
            if layout:
                buffer = np.fromstring(buffer, np.uint8)
                img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                img = resize_saving_ratio(
                    img, (layout.get('max_height'), layout.get('max_width'))
                )
                retval, buffer = cv2.imencode('.jpg', img)

            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # noinspection PyTypeChecker
            res['data'].append(
                {
                    'content': jpg_as_text,
                    'id': img_obj.id,
                    'y_pred': img_obj.y_pred,
                    'y': img_obj.y,
                    'score': round(img_obj.score, 2) if img_obj.score else None
                }
            )

        return res

    def detail_img_segment(
        self, filter: dict, options: PaginatorOptions = None
    ):
        res = {'data': []}
        res.update(filter)

        query = self.query(ReportImg).filter(
            ReportImg.task == filter['task']). \
            filter(ReportImg.group == filter['group'])

        if filter.get('y') is not None and filter.get('y_pred') is not None:
            query = query.filter(
                and_(
                    ReportImg.y == filter['y'],
                    ReportImg.y_pred == filter['y_pred']
                )
            )

        if filter.get('score_min') is not None:
            query = query.filter(
                or_(
                    ReportImg.score >= filter['score_min'],
                    ReportImg.score.__eq__(None)
                )
            )

        if filter.get('score_max') is not None:
            query = query.filter(
                or_(
                    ReportImg.score <= filter['score_max'],
                    ReportImg.score.__eq__(None)
                )
            )

        layout = filter.get('layout')

        if layout and layout.get('attrs'):
            for attr in layout['attrs']:
                field = getattr(ReportImg, attr['source'])
                if attr.get('equal') is not None:
                    query = query.filter(field == attr['equal'])
                if attr.get('greater') is not None:
                    query = query.filter(field >= attr['greater'])
                if attr.get('less') is not None:
                    query = query.filter(field <= attr['less'])

        res['total'] = query.count()

        query = self.paginator(query, options)
        img_objs = query.all()
        for img_obj in img_objs:
            buffer = img_obj.img
            if layout:
                buffer = np.fromstring(buffer, np.uint8)
                img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                img = resize_saving_ratio(
                    img, (layout.get('max_height'), layout.get('max_width'))
                )
                retval, buffer = cv2.imencode('.jpg', img)

            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # noinspection PyTypeChecker
            res['data'].append(
                {
                    'content': jpg_as_text,
                    'id': img_obj.id,
                    'y_pred': img_obj.y_pred,
                    'y': img_obj.y,
                    'score': round(img_obj.score, 2)
                    if img_obj.score is not None else None
                }
            )

        return res


__all__ = ['ReportImgProvider']
