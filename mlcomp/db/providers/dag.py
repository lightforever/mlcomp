from mlcomp.db.providers.base import *
from mlcomp.utils.misc import to_snake


class DagProvider(BaseDataProvider):
    model = Dag

    def get(self, options: PaginatorOptions = None):
        task_count = func.count(Task.id).label('task_count')
        last_activity = func.max(Task.last_activity).label('last_activity')
        task_status = []
        for e in TaskStatus:
            task_status.append(func.count(Task.status).filter(Task.status==e.value).label(e.name))

        query = self.query(Dag, task_count, last_activity, *task_status).join(Task).group_by(Dag.id)
        paginator = self.paginator(query, options) if options else query
        res = []
        rules = ('-tasks.dag_rel',)
        for dag, task_count, last_activity, *(task_status) in paginator.all():
            dag_dict = {k: v for k, v in dag.to_dict(rules=rules).items() if k not in ['tasks', 'config']}
            tasks = dag.tasks
            r = {
                'task_count': task_count,
                'last_activity': last_activity,
                **dag_dict
            }
            for t in tasks:
                r[to_snake(TaskStatus(t.status).name)] += 1

            r['task_statuses'] = [{'name': to_snake(e.name), 'count': s} for e, s in zip(TaskStatus, task_status)]
            r['last_activity'] = self.serializer.serialize_date(r['last_activity']) if r['last_activity'] else None
            res.append(r)
        return res


if __name__ == '__main__':
    DagProvider().get()
