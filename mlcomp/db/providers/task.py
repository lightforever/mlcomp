import datetime
from typing import List

from sqlalchemy import func, case
from sqlalchemy.orm import joinedload, aliased

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.db.enums import TaskType, DagType, TaskStatus
from mlcomp.utils.misc import to_snake, duration_format, now, parse_time
from mlcomp.db.models import Task, Project, Dag, TaskDependence, ReportTasks


class TaskProvider(BaseDataProvider):
    model = Task

    def _get_filter(self, query, filter: dict):
        if filter.get('dag'):
            query = query.filter(Task.dag == filter['dag'])

        if filter.get('name'):
            query = query.filter(Task.name.like(f'%{filter["name"]}%'))

        if filter.get('status'):
            status = [
                TaskStatus.from_name(k) for k, v in filter['status'].items()
                if v
            ]
            if len(status) > 0:
                query = query.filter(Task.status.in_(status))

        if filter.get('id_min'):
            query = query.filter(Task.id >= filter['id_min'])

        if filter.get('id_max'):
            query = query.filter(Task.id <= filter['id_max'])

        if filter.get('id'):
            query = query.filter(Task.id == filter['id'])

        if filter.get('project'):
            query = query.filter(Dag.project == filter['project'])

        if filter.get('created_min'):
            created_min = parse_time(filter['created_min'])
            query = query.filter(Dag.created >= created_min)
        if filter.get('created_max'):
            created_max = parse_time(filter['created_max'])
            query = query.filter(Dag.created <= created_max)
        if filter.get('last_activity_min'):
            last_activity_min = parse_time(filter['last_activity_min'])
            query = query.filter(Task.last_activity >= last_activity_min)
        if filter.get('last_activity_max'):
            last_activity_max = parse_time(filter['last_activity_max'])
            query = query.filter(Task.last_activity <= last_activity_max)
        if filter.get('report'):
            query = query.filter(Task.report is not None)
        if filter.get('parent'):
            query = query.filter(Task.parent == filter['parent'])

        types = filter.get('type', ['User', 'Train'])
        types = [TaskType.from_name(t) for t in types]
        query = query.filter(Task.type.in_(types))

        return query

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Task, Project.name). \
            join(Dag, Dag.id == Task.dag). \
            join(Project, Project.id == Dag.project). \
            options(joinedload(Task.dag_rel, innerjoin=True))

        query = self._get_filter(query, filter)

        total = query.count()
        paginator = self.paginator(query, options)
        res = []

        for p, project_name in paginator.all():
            if p.dag_rel is None:
                continue

            item = {**self.to_dict(p, rules=('-additional_info', ))}
            item['status'] = to_snake(TaskStatus(item['status']).name)
            item['type'] = to_snake(TaskType(item['type']).name)
            item['dag_rel']['project'] = {
                'id': item['dag_rel']['project'],
                'name': project_name
            }
            if p.started is None:
                delta = 0
            elif p.status == TaskStatus.InProgress.value:
                delta = (now() - p.started).total_seconds()
            else:
                finish = (p.finished or p.last_activity)
                delta = (finish - p.started).total_seconds()
            item['duration'] = duration_format(delta)
            if p.dag_rel is not None:
                res.append(item)

        if filter.get('report'):
            tasks_within_report = self.query(
                ReportTasks.task
            ).filter(ReportTasks.report == int(filter['report']))
            tasks_within_report = {t[0] for t in tasks_within_report}
            for r in res:
                r['report_full'] = r['id'] in tasks_within_report

        projects = self.query(Project.name, Project.id). \
            order_by(Project.id.desc()). \
            limit(20). \
            all()
        dags = self.query(Dag.name, Dag.id). \
            order_by(Dag.id.desc()). \
            limit(20). \
            all()
        projects = [{'name': name, 'id': id} for name, id in projects]
        dags = [{'name': name, 'id': id} for name, id in dags]

        dags_model = self.query(Dag.name, Dag.id, Dag.project). \
            filter(Dag.type == DagType.Pipe.value). \
            order_by(Dag.id.desc()). \
            all()

        dags_model_dict = []
        used_dag_names = set()

        for name, id, project in dags_model:
            if name in used_dag_names:
                continue

            dag = {'name': name, 'id': id, 'project': project}
            dags_model_dict.append(dag)
            used_dag_names.add(name)

        return {
            'total': total,
            'data': res,
            'projects': projects,
            'dags': dags,
            'dags_model': dags_model_dict
        }

    def last_tasks(self, min_time: datetime, status: int, joined_load=None):
        res = self.query(Task).filter(
            Task.finished >= min_time). \
            filter(Task.status == status)

        if joined_load is not None:
            for n in joined_load:
                res = res.options(joinedload(n, innerjoin=True))

        return res.all()

    def add_dependency(self, task_id: int, depend_id: int) -> None:
        self.add(TaskDependence(task_id=task_id, depend_id=depend_id))

    def by_id(self, id, options=None) -> Task:
        query = self.query(Task).filter(Task.id == id)
        if options:
            query = query.options(options)
        return query.one_or_none()

    def change_status(self, task, status: TaskStatus):
        if status == TaskStatus.InProgress:
            task.started = now()
        elif status in [
            TaskStatus.Failed, TaskStatus.Stopped, TaskStatus.Success
        ]:
            task.finished = now()

        task.status = status.value
        self.update()

    def by_status(
        self,
        *statuses: TaskStatus,
        task_docker_assigned: str = None,
        worker_index: int = None,
        computer_assigned: str = None
    ):
        statuses = [s.value for s in statuses]
        query = self.query(Task).filter(Task.status.in_(statuses)). \
            options(joinedload(Task.dag_rel, innerjoin=True))

        if task_docker_assigned:
            query = query.filter(Task.docker_assigned == task_docker_assigned)
        if worker_index is not None:
            query = query.filter(Task.worker_index == worker_index)
        if computer_assigned is not None:
            query = query.filter(Task.computer_assigned == computer_assigned)
        return query.all()

    def dependency_status(self, tasks: List[Task]):
        res = {t.id: set() for t in tasks}
        task_ids = [task.id for task in tasks]
        items = self.query(TaskDependence, Task). \
            filter(TaskDependence.task_id.in_(task_ids)). \
            join(Task, Task.id == TaskDependence.depend_id).all()
        for item, task in items:
            res[item.task_id].add(task.status)

        return res

    def update_last_activity(self, task: int):
        self.query(Task).filter(Task.id == task
                                ).update({'last_activity': now()})
        self.session.commit()

    def stop(self, id: int):
        task = self.by_id(id)
        self.change_status(task, TaskStatus.Stopped)

    def last_succeed_time(self):
        res = self.query(Task.finished). \
            filter(Task.status == TaskStatus.Success.value). \
            order_by(Task.finished.desc()). \
            first()
        return res[0] if res else None

    def by_dag(self, id: int):
        return self.query(Task).filter(Task.dag == id).order_by(Task.id).all()

    def parent_tasks_stats(self):
        task_parent = aliased(Task)
        task_child = aliased(Task)

        task_status = []
        for e in TaskStatus:
            task_status.append(
                func.sum(
                    case(whens=[(task_child.status == e.value, 1)],
                         else_=0).label(e.name)
                )
            )

        times = [func.min(task_child.started), func.max(task_child.finished)]

        parent_statuses = [
            TaskStatus.Queued.value, TaskStatus.InProgress.value
        ]

        query = self.query(task_parent, *times, *task_status). \
            filter(task_parent.status.in_(parent_statuses)). \
            join(task_child, task_parent.id == task_child.parent). \
            group_by(task_parent.id)

        res = []
        for task, started, finished, *(statuses) in query.all():
            res.append(
                [
                    task, started, finished,
                    {e: s
                     for e, s in zip(TaskStatus, statuses)}
                ]
            )

        return res

    def has_id(self, id: int):
        return self.query(Task).filter(Task.id == id).count() > 0

    def children(self, id: int, joined_load=None):
        res = self.query(Task).filter(Task.parent == id)
        res = res.order_by(Task.id)
        if joined_load is not None:
            for n in joined_load:
                res = res.options(joinedload(n, innerjoin=True))
        return res.all()

    def project(self, task_id: int):
        return self.query(Project).join(Dag).join(Task).filter(
            Task.id == task_id
        ).one()


__all__ = ['TaskProvider']
