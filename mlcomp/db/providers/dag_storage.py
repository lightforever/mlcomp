from mlcomp.db.models import DagStorage, File, DagLibrary
from mlcomp.db.providers.base import BaseDataProvider


class DagStorageProvider(BaseDataProvider):
    model = DagStorage

    def by_dag(self, dag: int):
        query = self.query(DagStorage, File).join(File, isouter=True). \
            filter(DagStorage.dag == dag). \
            order_by(DagStorage.path)
        return query.all()


class DagLibraryProvider(BaseDataProvider):
    model = DagLibrary

    def dag(self, dag: int):
        return self.query(DagLibrary.library, DagLibrary.version). \
            filter(DagLibrary.dag == dag).all()


__all__ = ['DagStorageProvider', 'DagLibraryProvider']
