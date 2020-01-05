import os
from typing import List

from sqlalchemy import func

from mlcomp import DATA_FOLDER, MODEL_FOLDER
from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import Project, Dag, Task
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.io import yaml_dump


class ProjectProvider(BaseDataProvider):
    model = Project

    def add_project(self, name: str,
                    class_names: dict = None,
                    ignore_folders: List[str] = None):
        class_names = class_names or {}
        ignore_folders = ignore_folders or []

        assert type(class_names) == dict, 'class_names type must be dict'
        assert isinstance(ignore_folders, list), \
            'ignore_folders type must be list'

        project = Project(name=name,
                          class_names=yaml_dump(class_names),
                          ignore_folders=yaml_dump(ignore_folders)
                          )
        project = self.session.add(project)

        os.makedirs(os.path.join(DATA_FOLDER, name), exist_ok=True)
        os.makedirs(os.path.join(MODEL_FOLDER, name), exist_ok=True)

        return project

    def edit_project(self, name: str,
                     class_names: dict,
                     ignore_folders: List[str]):
        assert type(class_names) == dict, 'class_names type must be dict'
        assert isinstance(ignore_folders, list), \
            'ignore_folders type must be list'

        project = self.by_name(name)
        project.class_names = yaml_dump(class_names)
        project.ignore_folders = yaml_dump(ignore_folders)
        self.commit()

    def get(self, filter: dict = None, options: PaginatorOptions = None):
        filter = filter or {}

        query = self.query(Project,
                           func.count(Dag.id),
                           func.max(Task.last_activity)). \
            join(Dag, Dag.project == Project.id, isouter=True). \
            join(Task, isouter=True). \
            group_by(Project.id)

        if filter.get('name'):
            query = query.filter(Project.name.like(f'%{filter["name"]}%'))

        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p, dag_count, last_activity in paginator.all():
            last_activity = self.serializer.serialize_datetime(last_activity) \
                if last_activity else None

            file_size, img_size = self.query(func.sum(Dag.file_size),
                                             func.sum(Dag.img_size)).filter(
                Dag.project == p.id).one()

            res.append(
                {
                    'dag_count': dag_count,
                    'last_activity': last_activity,
                    'img_size': int(img_size or 0),
                    'file_size': int(file_size or 0),
                    'id': p.id,
                    'name': p.name,
                    'ignore_folders': p.ignore_folders,
                    'class_names': p.class_names
                })
        return {'total': total, 'data': res}

    def all_last_activity(self):
        query = self.query(Project,
                           func.max(Task.last_activity)). \
            join(Dag, Dag.project == Project.id, isouter=True). \
            join(Task, isouter=True). \
            group_by(Project.id)

        res = query.all()
        for p, last_activity in res:
            p.last_activity = last_activity
        return [r[0] for r in res]

    def by_name(self, name: str):
        return self.query(Project).filter(Project.name == name).first()


__all__ = ['ProjectProvider']
