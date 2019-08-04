import base64
import pickle

from sqlalchemy import and_

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import Project, Dag, ReportImg, Task
from mlcomp.db.providers.base import BaseDataProvider
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
            filter(ReportImg.part == filter['part']). \
            filter(ReportImg.group == filter['group'] + '_confusion'). \
            filter(ReportImg.epoch == filter['epoch']).first()
        if confusion:
            confusion = pickle.loads(confusion[0])['data'].tolist()
            res['confusion'] = {'data': confusion}

        res.update(filter)

        query = self.query(ReportImg).filter(
            ReportImg.task == filter['task']).filter(
            ReportImg.epoch == filter['epoch']). \
            filter(ReportImg.group == filter['group']).filter(
            ReportImg.part == filter['part'])

        if filter.get('y') is not None and filter.get('y_pred') is not None:
            query = query.filter(
                and_(
                    ReportImg.y == filter['y'],
                    ReportImg.y_pred == filter['y_pred']
                )
            )

        if filter.get('metric_diff_min') is not None:
            query = query.filter(
                ReportImg.metric_diff >= filter['metric_diff_min']
            )
        if filter.get('metric_diff_max') is not None:
            query = query.filter(
                ReportImg.metric_diff <= filter['metric_diff_max']
            )

        project = self.query(Project).join(Dag).join(Task).filter(
            Task.id == filter['task']
        ).first()
        class_names = yaml_load(project.class_names)

        res['total'] = query.count()
        if 'default' in class_names:
            res['class_names'] = class_names['default']
        else:
            res['class_names'] = [str(i) for i in confusion.shape[0]]

        query = self.paginator(query, options)
        img_objs = query.all()
        for img_obj in img_objs:
            img = pickle.loads(img_obj.img)
            # noinspection PyTypeChecker
            res['data'].append(
                {
                    'content': base64.b64encode(img['img']).decode('utf-8'),
                    'id': img_obj.id,
                    'y_pred': img_obj.y_pred,
                    'y': img_obj.y,
                    'metric_diff': round(img_obj.metric_diff, 2)
                }
            )

        return res


__all__ = ['ReportImgProvider']
