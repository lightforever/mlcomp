from mlcomp.db.providers.base import *
from mlcomp.utils.misc import to_snake, duration_format


class DagProvider(BaseDataProvider):
    model = Dag

    def get(self, filter: dict, options: PaginatorOptions = None):
        task_status = []
        for e in TaskStatus:
            task_status.append(
                func.count(Task.status).filter(Task.status == e.value).label(
                    e.name))

        last_activity = func.max(Task.last_activity).label('last_activity')
        funcs = [
            func.count(Task.id).label('task_count'),
            last_activity,
            func.min(Task.started).label('started'),
            func.min(Task.finished).label('finished')
        ]

        query = self.query(Dag, *funcs, *task_status)
        if filter.get('project'):
            query = query.filter(Dag.project == filter['project'])
        if filter.get('name'):
            query = query.filter(Dag.name.like(f'%{filter["name"]}%'))
        if filter.get('id'):
            query = query.filter(Dag.id == int(filter['id']))

        if filter.get('created_min'):
            query = query.filter(Dag.created >= filter['created_min'])
        if filter.get('created_max'):
            query = query.filter(Dag.created <= filter['created_max'])
        if filter.get('last_activity_min'):
            query = query.having(last_activity >= filter['last_activity_min'])
        if filter.get('last_activity_max'):
            query = query.having(last_activity <= filter['last_activity_max'])

        status_clauses = []
        for agg, e in zip(task_status, TaskStatus):
            if filter['status'].get(to_snake(e.name)):
                status_clauses.append(agg > 0)
        if len(status_clauses) > 0:
            query = query.having(or_(*status_clauses))

        query = query.join(Task, isouter=True).group_by(Dag.id)
        total = query.count()
        paginator = self.paginator(query, options) if options else query
        res = []
        rules = ('-tasks.dag_rel',)
        for dag, task_count, last_activity, \
            started, finished, *(task_status) in paginator.all():
            # noinspection PyDictCreation
            r = {
                'task_count': task_count,
                'last_activity': last_activity,
                'started': started,
                'finished': finished,
                **{k: v for k, v in self.to_dict(dag, rules=rules).items() if
                   k not in ['tasks', 'config']}
            }

            r['task_statuses'] = [{'name': to_snake(e.name), 'count': s} for
                                  e, s in zip(TaskStatus, task_status)]
            r['last_activity'] = self.serializer.serialize_datetime(
                r['last_activity']) if r['last_activity'] else None
            r['started'] = self.serializer.serialize_datetime(r['started']) if \
                r['started'] else None
            r['finished'] = self.serializer.serialize_datetime(
                r['finished']) if r['finished'] else None

            if task_status[TaskStatus.InProgress.value] > 0:
                delta = (now() - started).total_seconds()
            elif sum(task_status[
                     TaskStatus.InProgress.value:]) == 0 or not started:
                delta = 0
            else:
                delta = (finished - started).total_seconds()

            r['duration'] = duration_format(delta)
            res.append(r)

        if filter.get('report'):
            dag_ids = [r['id'] for r in res]
            tasks_dags = self.query(Task.id, Task.dag).filter(
                Task.dag.in_(dag_ids)).all()
            tasks_within_report = self.query(ReportTasks.task).filter(
                ReportTasks.report == int(filter['report']))
            tasks_within_report = {t[0] for t in tasks_within_report}
            dags_not_full_included = {d for t, d in tasks_dags if
                                      t not in tasks_within_report}
            for r in res:
                r['report_full'] = r['id'] not in dags_not_full_included
        projects = self.query(Project.name, Project.id).order_by(
            Project.id.desc()).limit(20).all()
        projects = [{'name': name, 'id': id} for name, id in projects]
        return {'total': total, 'data': res, 'projects': projects}

    def config(self, id: int):
        return self.by_id(id).config

    def duration(self, t: Task):
        if not t.started:
            return duration_format(0)
        delta = ((
                     t.finished if t.status > TaskStatus.InProgress.value else now()) - t.started).total_seconds()
        return duration_format(delta)

    def graph(self, id: int):
        tasks = self.query(Task).filter(Task.dag == id).all()
        task_ids = [t.id for t in tasks]
        dep = self.query(TaskDependence).filter(
            TaskDependence.task_id.in_(task_ids)).all()
        task_by_id = {t.id: t for t in tasks}

        def label(t: Task):
            res = [t.executor]
            if t.status >= TaskStatus.InProgress.value:
                res.append(self.duration(t))
                res.append(f'{(t.current_step or 1)}/{t.steps}')
            return '\n'.join(res)

        nodes = [
            {
                'id': t.id,
                'label': label(t),
                'status': to_snake(TaskStatus(t.status).name)
            } for t in tasks]
        edges = [
            {'from': d.depend_id, 'to': d.task_id, 'status': to_snake(
                TaskStatus(task_by_id[d.depend_id].status).name)}
            for d in dep]
        return {'nodes': nodes, 'edges': edges}


if __name__ == '__main__':
    DagProvider().graph(39)
