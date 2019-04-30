from mlcomp.db.providers.base import *


class DagStorageProvider(BaseDataProvider):
    def by_dag(self, dag: int):
        return self.query(DagStorage, File).join(File, isouter=True).filter(Dag.id == dag).all()
