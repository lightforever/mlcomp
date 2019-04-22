from mlcomp.db.providers.base import *

class TaskProvider(BaseDataProvider):
    def add(self, task: Task):
        self.handler.insert(task)

    def get(self, options: PaginatorOptions):
        query = self.session.query(Task)
        return self.paginator(query, options).all()