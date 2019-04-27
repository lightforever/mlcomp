from mlcomp.db.providers.base import *

class TaskProvider(BaseDataProvider):
    def get(self, options: PaginatorOptions):
        query = self.session.query(Task)
        return self.paginator(query, options).all()

    def add_dependency(self, task_id:int, depend_id: int):
        self.add(TaskDependence(task_id=task_id, depend_id=depend_id))