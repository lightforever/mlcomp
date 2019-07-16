import pickle

from sqlalchemy import func

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import Project, Dag, Task
from mlcomp.db.providers.base import *


class ProjectProvider(BaseDataProvider):
    model = Project

    def add(self, name: str, class_names: dict):
        project = Project(name=name, class_names=pickle.dumps(class_names))
        self.session.add(project)

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Project,
                           func.count(Dag.id),
                           func.max(Task.last_activity),
                           func.sum(Dag.img_size),
                           func.sum(Dag.file_size)). \
            join(Dag, Dag.project == Project.id, isouter=True).\
            join(Task, isouter=True).\
            group_by(Project.id)

        if filter.get('name'):
            query = query.filter(Project.name.like(f'%{filter["name"]}%'))

        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p, \
            dag_count, \
            last_activity, \
            img_size, \
            file_size \
                in paginator.all():
            last_activity = self.serializer.serialize_datetime(last_activity) \
                if last_activity else None

            res.append(
                {
                    'dag_count': dag_count,
                    'last_activity': last_activity,
                    'img_size': int(img_size or 0),
                    'file_size': int(file_size or 0),
                    'id': p.id,
                    'name': p.name

                })
        return {'total': total, 'data': res}

    def by_name(self, name: str):
        return self.query(Project).filter(Project.name == name).first()


__all__ = ['ProjectProvider']