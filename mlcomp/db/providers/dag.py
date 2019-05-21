from collections import OrderedDict

from mlcomp.db.providers.base import *
from mlcomp.utils.misc import to_snake


class DagProvider(BaseDataProvider):
    model = Dag

    def get(self, filter: dict, options: PaginatorOptions = None):
        task_status = []
        for e in TaskStatus:
            task_status.append(func.count(Task.status).filter(Task.status == e.value).label(e.name))

        funcs = [
            func.count(Task.id).label('task_count'),
            func.max(Task.last_activity).label('last_activity'),
            func.min(Task.started).label('started'),
            func.min(Task.finished).label('finished')
        ]

        query = self.query(Dag, *funcs, *task_status)
        if filter.get('project'):
            query = query.filter(Dag.project == filter['project'])
        if filter.get('name'):
            query = query.filter(Task.name.like(f'%{filter["name"]}%'))
        if filter.get('id'):
            query = query.filter(Dag.id == int(filter['id']))

        query = query.join(Task).group_by(Dag.id)
        total = query.count()
        paginator = self.paginator(query, options) if options else query
        res = []
        rules = ('-tasks.dag_rel',)
        for dag, task_count, last_activity, started, finished, *(task_status) in paginator.all():
            # noinspection PyDictCreation
            r = {
                'task_count': task_count,
                'last_activity': last_activity,
                'started': started,
                'finished': finished,
                **{k: v for k, v in dag.to_dict(rules=rules).items() if k not in ['tasks', 'config']}
            }

            r['task_statuses'] = [{'name': to_snake(e.name), 'count': s} for e, s in zip(TaskStatus, task_status)]
            r['last_activity'] = self.serializer.serialize_date(r['last_activity']) if r['last_activity'] else None
            r['started'] = self.serializer.serialize_date(r['started']) if r['started'] else None
            r['finished'] = self.serializer.serialize_date(r['finished']) if r['finished'] else None
            res.append(r)
        return {'total': total, 'data': res}

    def config(self, id: int):
        return self.by_id(id).config

    def graph(self, id: int):
        tasks = self.query(Task).filter(Task.dag == id).all()
        task_ids = [t.id for t in tasks]
        dep = self.query(TaskDependence).filter(TaskDependence.task_id.in_(task_ids)).all()
        task_by_id = {t.id: t for t in tasks}

        def label(t: Task):
            res = [t.executor]
            if t.status >= TaskStatus.InProgress.value:
                delta = t.last_activity - t.started
                res.append(str(delta).split('.')[0])
                res.append(f'{t.current_step or 0}/{t.steps}')
            return '\n'.join(res)

        nodes = [
            {
                'id': t.id,
                'label': label(t),
                'status': to_snake(TaskStatus(t.status).name)
            } for t in tasks]
        edges = [
            {'from': d.depend_id, 'to': d.task_id, 'status': to_snake(TaskStatus(task_by_id[d.depend_id].status).name)}
            for d in dep]
        return {'nodes': nodes, 'edges': edges}


if __name__ == '__main__':
    DagProvider().graph(39)
