from mlcomp.db.models import Log, Step, Task, Computer
from mlcomp.db.core import PaginatorOptions
from mlcomp.db.enums import ComponentType
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.utils.misc import log_name, to_snake


class LogProvider(BaseDataProvider):
    model = Log

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Log, Step, Task). \
            join(Step, Step.id == Log.step, isouter=True). \
            join(Task, Task.id == Log.task, isouter=True)

        if filter.get('message'):
            query = query.filter(Log.message.contains(filter['message']))

        if filter.get('dag'):
            query = query.filter(Task.dag == filter['dag'])

        if filter.get('task'):
            child_tasks = self.query(Task.id
                                     ).filter(Task.parent == filter['task']
                                              ).all()
            child_tasks = [c[0] for c in child_tasks]
            child_tasks.append(filter['task'])

            query = query.filter(Task.id.in_(child_tasks))

        if len(filter.get('components', [])) > 0:
            query = query.filter(Log.component.in_(filter['components']))

        if filter.get('computer'):
            query = query.filter(Computer.name == filter['computer'])

        if len(filter.get('levels', [])) > 0:
            query = query.filter(Log.level.in_(filter['levels']))

        if filter.get('task_name'):
            query = query.filter(Task.name.like(f'%{filter["task_name"]}%'))

        if filter.get('step_name'):
            query = query.filter(Step.name.like(f'%{filter["step_name"]}%'))

        if filter.get('step'):
            query = query.filter(Step.id == filter['step'])

        total = query.count()
        data = []
        for log, step, task in self.paginator(query, options):
            item = {
                'id': log.id,
                'message': log.message.split('\n'),
                'module': log.module,
                'line': log.line,
                'time': self.serializer.serialize_datetime(log.time),
                'level': log_name(log.level),
                'component': to_snake(ComponentType(log.component).name),
                'computer': log.computer,
                'step': self.to_dict(step) if step else None,
                'task': self.to_dict(task, rules=('-additional_info', ))
                if task else None
            }
            data.append(item)

        return {'total': total, 'data': data}

    def last(self, count: int, dag: int = None, task: int = None):
        query = self.query(Log, Task.id).outerjoin(Task)
        if dag is not None:
            query = query.filter(Task.dag == dag)
        if task is not None:
            query = query.filter(Task.id == task)
        return query.order_by(Log.id.desc()).limit(count).all()


__all__ = ['LogProvider']
