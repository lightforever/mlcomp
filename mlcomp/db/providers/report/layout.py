import pickle

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import ReportLayout
from mlcomp.db.providers import BaseDataProvider
from mlcomp.utils.misc import now


class ReportLayoutProvider(BaseDataProvider):
    model = ReportLayout

    def get(self, filter: dict = None, options: PaginatorOptions = None):
        filter = filter or {}

        query = self.query(ReportLayout)
        total = query.count()
        paginator = self.paginator(query, options)

        res = []
        for item in paginator.all():
            res.append(self.to_dict(item))

        return {'total': total, 'data': res}

    def by_name(self, name: str):
        return self.query(ReportLayout).filter(ReportLayout.name == name).one()

    def add_item(self, k: str, v: dict):
        self.add(
            ReportLayout(content=pickle.dumps(v), name=k, last_modified=now()))

    def all(self):
        return {s.name: pickle.loads(s.content) for s in
                self.query(ReportLayout).all()}

    def change(self, k: str, v: dict):
        self.query(ReportLayout).filter(ReportLayout.name == k).update(
            {'last_modified': now(), 'content': pickle.dumps(v)})