from typing import List

from sqlalchemy.orm import joinedload

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.providers.base import BaseDataProvider, ReportTasks
from mlcomp.db.enums import TaskType, DagType, TaskStatus
from mlcomp.utils.misc import to_snake, duration_format, now
from mlcomp.utils.config import Config
from mlcomp.db.models import Task, Project, Dag, TaskDependence


class TaskProvider(BaseDataProvider):
    model = Task

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Task, Project.name). \
            options(joinedload(Task.dag_rel))

        if filter.get('dag'):
            query = query.filter(Task.dag == filter['dag'])

        if filter.get('name'):
            query = query.filter(Task.name.like(f'%{filter["name"]}%'))

        if filter.get('status'):
            status = [TaskStatus.from_name(k) for k, v in
                      filter['status'].items() if v]
            if len(status) > 0:
                query = query.filter(Task.status.in_(status))

        if filter.get('id'):
            query = query.filter(Task.id == filter['id'])

        if filter.get('project'):
            query = query.filter(Dag.project == filter['project'])

        if filter.get('created_min'):
            query = query.filter(Dag.created >= filter['created_min'])
        if filter.get('created_max'):
            query = query.filter(Dag.created <= filter['created_max'])
        if filter.get('last_activity_min'):
            query = query.filter(
                Task.last_activity >= filter['last_activity_min'])
        if filter.get('last_activity_max'):
            query = query.filter(
                Task.last_activity <= filter['last_activity_max'])

        total = query.count()
        paginator = self.paginator(query, options)
        res = []
        for p, project_name in paginator.all():
            item = {**self.to_dict(p, rules=('-additional_info',))}
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
            tasks_within_report = self.query(ReportTasks.task).filter(
                ReportTasks.report == int(filter['report']))
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

        dags_model = self.query(Dag.name, Dag.id, Dag.config). \
            filter(Dag.type == DagType.Pipe.value). \
            order_by(Dag.id.desc()). \
            all()
        dags_model_dict = []
        for name, id, config in dags_model:
            config = Config.from_yaml(config)
            slots = []
            for pipe in config['pipes'].values():
                for k, v in pipe.items():
                    if 'slot' in v:
                        slots.append(v['slot'])
                    elif 'slots' in v:
                        slots.extend(v['slots'])

            dag = {'name': name,
                   'id': id,
                   'slots': slots,
                   'interfaces': config['interfaces']
                   }
            dags_model_dict.append(dag)
        return {'total': total,
                'data': res,
                'projects': projects,
                'dags': dags,
                'dags_model': dags_model_dict
                }

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
        elif status in [TaskStatus.Failed,
                        TaskStatus.Stopped,
                        TaskStatus.Success]:
            task.finished = now()

        task.status = status.value
        self.update()

    def by_status(self, status: TaskStatus, docker_img: str = None,
                  worker_index: int = None):
        query = self.query(Task).filter(Task.status == status.value). \
            options(joinedload(Task.dag_rel))

        if docker_img:
            query = query.join(Dag).filter(Dag.docker_img == docker_img)
        if worker_index is not None:
            query = query.filter(Task.worker_index == worker_index)
        return query.all()

    def dependency_status(self, tasks: List[Task]):
        res = {t.id: [] for t in tasks}
        task_ids = [task.id for task in tasks]
        items = self.query(TaskDependence, Task). \
            filter(TaskDependence.task_id.in_(task_ids)). \
            join(Task, Task.id == TaskDependence.depend_id).all()
        for item, task in items:
            res[item.task_id].append(task.status)

        return res

    def update_last_activity(self, task: int):
        self.query(Task).filter(Task.id == task).update(
            {'last_activity': now()})
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


__all__ = ['TaskProvider']
