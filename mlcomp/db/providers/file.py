from mlcomp.db.models import File, Dag
from mlcomp.db.providers.base import *


class FileProvider(BaseDataProvider):
    model = File

    def hashs(self, project: int):
        return {
            obj[0]: obj[1] for obj in self.query(File.md5, File.id).
                filter(File.project == project).all()
        }

    def remove(self, filter: dict):
        query = self.query(File)
        if filter.get('dag'):
            query = query.filter(File.dag == filter['dag'])
        if filter.get('project'):
            query = query.filter(File.project == filter['project'])
        query.delete(synchronize_session=False)
        self.session.commit()

        query = self.query(Dag)
        if filter.get('dag'):
            query.filter(Dag.id == filter['dag']).update({'file_size': 0})

        if filter.get('project'):
            query.filter(Dag.project == filter['project']).\
                update({'file_size': 0})

        self.session.commit()


__all__ = ['FileProvider']
