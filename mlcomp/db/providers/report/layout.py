from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import ReportLayout
from mlcomp.db.providers import BaseDataProvider
from mlcomp.db.report_info import ReportLayoutInfo
from mlcomp.utils.misc import now, yaml_dump, yaml_load


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
            ReportLayout(content=yaml_dump(v),
                         name=k,
                         last_modified=now()))

    def all(self):
        res = {s.name: yaml_load(s.content) for s in
               self.query(ReportLayout).all()}

        for k, v in res.items():
            res[k] = ReportLayoutInfo.union_layouts(k, res)
        return res

    def change(self, k: str, v: dict):
        self.query(ReportLayout).filter(ReportLayout.name == k).update(
            {
                'last_modified': now(),
                'content': yaml_dump(v)
            })


__all__ = ['ReportLayoutProvider']