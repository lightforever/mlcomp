from mlcomp.db.providers.base import *


class ModelProvider(BaseDataProvider):
    model = Model

    def get(self, filter, options: PaginatorOptions):
        query = self.query(Model)

        if filter.get('project'):
            query = query.filter(Model.project == filter['project'])
        if filter.get('name'):
            query = query.filter(Model.name.like(f'%{filter["name"]}%'))

        total = query.count()
        paginator = self.paginator(query, options) if options else query
        res = []
        for model in paginator.all():
            res.append(model.to_dict())

        return {'total': total, 'data': res}
