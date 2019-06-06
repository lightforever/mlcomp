from mlcomp.db.providers.base import *
import pickle


class ProjectProvider(BaseDataProvider):
    model = Project

    def add(self, name: str, class_names: dict):
        project = Project(name=name, class_names=pickle.dumps(class_names))
        self.session.add(project)

    def img_size(self, id: int):
        res = self.session.execute('SELECT sum(octet_length(t.*::text)) FROM report_img as t where project=:p',
                                   {'p': id}).fetchone()[0]
        return 0 if not res else int(res / 2)

    def file_size(self, id: int):
        res = self.session.execute('SELECT sum(octet_length(t.*::text)) FROM file as t where project=:p',
                                   {'p': id}).fetchone()[0]
        return 0 if not res else int(res / 2)

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Project, func.count(Dag.id), func.max(Task.last_activity)). \
            join(Dag, Dag.project == Project.id, isouter=True).join(Task, isouter=True).group_by(Project.id)
        if filter.get('name'):
            query = query.filter(Project.name.like(f'%{filter["name"]}%'))

        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p, dag_count, last_activity in paginator.all():
            res.append(
                {
                    'dag_count': dag_count,
                    'last_activity': self.serializer.serialize_datetime(last_activity) if last_activity else None,
                    'img_size': self.img_size(p.id),
                    'file_size': self.file_size(p.id),
                    'id': p.id,
                    'name': p.name

                })
        return {'total': total, 'data': res}

    def by_name(self, name: str):
        return self.query(Project).filter(Project.name == name).first()
