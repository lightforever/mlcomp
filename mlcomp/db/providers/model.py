from collections import defaultdict

from sqlalchemy.orm import joinedload

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.enums import DagType
from mlcomp.db.models import Model, Dag, Project
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.config import Config
from mlcomp.utils.misc import parse_time


class ModelProvider(BaseDataProvider):
    model = Model

    def get(self, filter, options: PaginatorOptions):
        query = self.query(Model). \
            options(joinedload(Model.dag_rel, innerjoin=True)). \
            options(joinedload(Model.project_rel, innerjoin=True))

        if filter.get('project'):
            query = query.filter(Model.project == filter['project'])
        if filter.get('name'):
            query = query.filter(Model.name.like(f'%{filter["name"]}%'))

        if filter.get('created_min'):
            created_min = parse_time(filter['created_min'])
            query = query.filter(Model.created >= created_min)
        if filter.get('created_max'):
            created_max = parse_time(filter['created_max'])
            query = query.filter(Model.created <= created_max)

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
            order_by(Dag.id.desc()). \
            all()

        dags_by_project = defaultdict(list)
        used_dag_names = set()

        for dag in models_dags:
            if dag.name in used_dag_names:
                continue

            config = Config.from_yaml(dag.config)
            slots = []
            for pipe in config['pipes'].values():
                for k, v in pipe.items():
                    if 'slot' in v:
                        if v['slot'] not in slots:
                            slots.append(v['slot'])
                    elif 'slots' in v:
                        for slot in v['slots']:
                            if slot not in slots:
                                slots.append(slot)

            d = {'name': dag.name,
                 'id': dag.id,
                 'slots': slots,
                 'interfaces': list(config['interfaces']),
                 'pipes': list(config['pipes'])
                 }

            dags_by_project[dag.project].append(d)
            used_dag_names.add(dag.name)

        for row in res:
            row['dags'] = dags_by_project[row['project']]

        projects = self.query(Project.name, Project.id). \
            order_by(Project.id.desc()). \
            limit(20). \
            all()
        projects = [{'name': name, 'id': id} for name, id in projects]
        return {'total': total, 'data': res, 'projects': projects}

    def change_dag(self, project: int, name: str, to: int):
        ids = self.query(Model.id). \
            join(Dag). \
            filter(Model.project == project). \
            filter(Dag.name == name). \
            filter(Dag.type == DagType.Pipe.value). \
            all()

        ids = [id[0] for id in ids]

        self.query(Model). \
            filter(Model.id.in_(ids)). \
            update({'dag': to}, synchronize_session=False)
        self.commit()


__all__ = ['ModelProvider']
