from mlcomp.db.providers.base import *

class ProjectProvider(BaseDataProvider):
    model = Project

    def add(self, name: str):
        project = Project(name=name)
        self.session.add(project)

    def get(self, options: PaginatorOptions):
        query = self.query(Project, func.count(Dag.id), func.max(Task.last_activity)).join(Dag).join(Task).group_by(Project.id)
        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p, dag_count, last_activity in paginator.all():
            res.append({'dag_count': dag_count, 'last_activity': self.serializer.serialize_date(last_activity), **p.to_dict()})
        return {'total': total, 'data': res}

    def by_name(self, name: str):
        return self.query(Project).filter(Project.name==name).first()