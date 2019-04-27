from mlcomp.db.providers.base import *

class TaskProvider(BaseDataProvider):
    def get(self, options: PaginatorOptions):
        query = self.session.query(Task)
        return self.paginator(query, options).all()