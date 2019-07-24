from mlcomp.db.models import Auxiliary
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.io import yaml_load


class AuxiliaryProvider(BaseDataProvider):
    model = Auxiliary

    def get(self):
        query = self.query(self.model)
        res = dict()
        for r in query.all():
            res[r.name] = yaml_load(r.data)
        return res


__all__ = ['AuxiliaryProvider']
