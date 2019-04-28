from mlcomp.db.providers.base import *
from typing import List
from collections import defaultdict

class TaskProvider(BaseDataProvider):
    def get(self, options: PaginatorOptions):
        query = self.session.query(Task)
        return self.paginator(query, options).all()

    def add_dependency(self, task_id: int, depend_id: int) -> None:
        self.add(TaskDependence(task_id=task_id, depend_id=depend_id))

    def by_id(self, id) -> Task:
        return self.query(Task).filter(Task.id == id).first()

    def change_status(self, task, status: TaskStatus):
        task.status = status.value
        if status == TaskStatus.InProgress:
            task.started = now()
        elif status in [TaskStatus.Failed, TaskStatus.Stopped, TaskStatus.Success]:
            task.finished = now()

        self.session.update()

    def by_status(self, status: TaskStatus):
        return self.query(Task).filter(Task.status == status.value).all()

    def dependency_status(self, tasks: List[Task]):
        res = {t.id: [] for t in tasks}
        task_ids = [task.id for task in tasks]
        items = self.query(TaskDependence, Task).filter(TaskDependence.task_id.in_(task_ids)).\
            join(Task, Task.id==TaskDependence.depend_id).all()
        for item, task in items:
            res[item.task_id].append(task.status)

        return res
