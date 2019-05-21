from mlcomp.db.providers.base import *
from typing import List
from sqlalchemy.orm.attributes import flag_modified

from utils.misc import to_snake


class TaskProvider(BaseDataProvider):
    model = Task

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Task).options(joinedload(Task.dag_rel))
        if filter.get('dag'):
            query = query.filter(Task.dag==filter['dag'])

        if filter.get('name'):
            query = query.filter(Project.name.like(f'%{filter["name"]}%'))

        if filter.get('status'):
            query = query.filter(Task.status==TaskStatus.from_name(filter['status']))

        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p in paginator.all():
            item = {**p.to_dict()}
            item['status'] = to_snake(TaskStatus(item['status']).name)
            res.append(item)
        return {'total': total, 'data': res}

    def add_dependency(self, task_id: int, depend_id: int) -> None:
        self.add(TaskDependence(task_id=task_id, depend_id=depend_id))

    def by_id(self, id, options=None) -> Task:
        query = self.query(Task).filter(Task.id == id)
        if options:
            query = query.options(options)
        return query.first()

    def change_status(self, task, status: TaskStatus):
        if status == TaskStatus.InProgress:
            task.started = now()
        elif status in [TaskStatus.Failed, TaskStatus.Stopped, TaskStatus.Success]:
            task.finished = now()

        task.status = status.value
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

    def update_last_activity(self, task:int):
        self.query(Task).filter(Task.id==task).update({'last_activity': now()})

    def stop(self, id:int):
        task = self.by_id(id)
        self.change_status(task, TaskStatus.Stopped)