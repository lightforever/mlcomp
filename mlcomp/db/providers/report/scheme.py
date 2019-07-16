import pickle

from mlcomp.db.models import ReportScheme
from mlcomp.db.providers import BaseDataProvider
from mlcomp.utils.misc import now


class ReportSchemeProvider(BaseDataProvider):
    model = ReportScheme

    def by_name(self, name: str):
        return self.query(ReportScheme).filter(ReportScheme.name == name).one()

    def add_item(self, k: str, v: dict):
        self.add(
            ReportScheme(content=pickle.dumps(v), name=k, last_modified=now()))

    def all(self):
        return {s.name: pickle.loads(s.content) for s in
                self.query(ReportScheme).all()}

    def change(self, k: str, v: dict):
        self.query(ReportScheme).filter(ReportScheme.name == k).update(
            {'last_modified': now(), 'content': pickle.dumps(v)})