from collections import defaultdict

from sqlalchemy.orm import joinedload

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.enums import DagType
from mlcomp.db.models import Model, Dag, Project
from mlcomp.db.providers.base import *
from mlcomp.utils.config import Config


class ModelProvider(BaseDataProvider):
    model = Model

    def get(self, filter, options: PaginatorOptions):
        query = self.query(Model).\
            options(joinedload(Model.dag_rel)).\
            options(joinedload(Model.project_rel))

        if filter.get('project'):
            query = query.filter(Model.project == filter['project'])
        if filter.get('name'):
            query = query.filter(Model.name.like(f'%{filter["name"]}%'))

        if filter.get('created_min'):
            query = query.filter(Model.created >= filter['created_min'])
        if filter.get('created_max'):
            query = query.filter(Model.created <= filter['created_max'])

        total = query.count()
        paginator = self.paginator(query, options) if options else query
        res = []
        models = paginator.all()
        models_projects = set()
        for model in models:
            row = self.to_dict(model, rules=('-project_rel.class_names',))
            res.append(row)
            models_projects.add(model.project)

        models_dags = self.query(Dag). \
            filter(Dag.type == DagType.Pipe.value). \
            filter(Dag.project.in_(list(models_projects))). \
            all()
        dags_by_project = defaultdict(list)
        for dag in models_dags:
            config = Config.from_yaml(dag.config)
            slots = []
            for pipe in config['pipes'].values():
                for k, v in pipe.items():
                    if 'slot' in v:
                        slots.append(v['slot'])
                    elif 'slots' in v:
                        slots.extend(v['slots'])

            d = {'name': dag.name,
                 'id': dag.id,
                 'slots': slots,
                 'interfaces': config['interfaces'],
                 'pipes': list(config['pipes'])
                 }

            dags_by_project[dag.project].append(d)

        for row in res:
            row['dags'] = dags_by_project[row['project']]

        projects = self.query(Project.name, Project.id). \
            order_by(Project.id.desc()). \
            limit(20). \
            all()
        projects = [{'name': name, 'id': id} for name, id in projects]
        return {'total': total, 'data': res, 'projects': projects}


__all__ = ['ModelProvider']