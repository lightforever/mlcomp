from mlcomp.db.providers.base import *
from typing import List
from mlcomp.utils.misc import to_snake, duration_format


class TaskProvider(BaseDataProvider):
    model = Task

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Task).options(joinedload(Task.dag_rel))
        if filter.get('dag'):
            query = query.filter(Task.dag == filter['dag'])

        if filter.get('name'):
            query = query.filter(Project.name.like(f'%{filter["name"]}%'))

        if filter.get('status'):
            query = query.filter(Task.status == TaskStatus.from_name(filter['status']))

        if filter.get('id'):
            query = query.filter(Task.id == filter['id'])

        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p in paginator.all():
            item = {**self.to_dict(p)}
            item['status'] = to_snake(TaskStatus(item['status']).name)
            if p.started is None:
                delta = 0
            elif p.status == TaskStatus.InProgress.value:
                delta = (now() - p.started).total_seconds()
            else:
                delta = ((p.finished or p.last_activity) - p.started).total_seconds()
            item['duration'] = duration_format(delta)
            if p.dag_rel is not None:
                res.append(item)

        if filter.get('report'):
            tasks_within_report = self.query(ReportTasks.task).filter(ReportTasks.report == int(filter['report']))
            tasks_within_report = {t[0] for t in tasks_within_report}
            for r in res:
                r['report_full'] = r['id'] in tasks_within_report

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
        self.update()

    def by_status(self, status: TaskStatus):
        return self.query(Task).filter(Task.status == status.value).options(joinedload(Task.dag_rel)).all()

    def dependency_status(self, tasks: List[Task]):
        res = {t.id: [] for t in tasks}
        task_ids = [task.id for task in tasks]
        items = self.query(TaskDependence, Task).filter(TaskDependence.task_id.in_(task_ids)). \
            join(Task, Task.id == TaskDependence.depend_id).all()
        for item, task in items:
            res[item.task_id].append(task.status)

        return res

    def update_last_activity(self, task: int):
        self.query(Task).filter(Task.id == task).update({'last_activity': now()})
        self.session.commit()

    def stop(self, id: int):
        task = self.by_id(id)
        self.change_status(task, TaskStatus.Stopped)
