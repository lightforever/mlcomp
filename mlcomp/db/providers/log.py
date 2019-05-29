from mlcomp.db.enums import ComponentType
from mlcomp.db.providers.base import *
from mlcomp.utils.misc import log_name, to_snake


class LogProvider(BaseDataProvider):
    model = Log

    def get(self, filter: dict, options: PaginatorOptions):
        query = self.query(Log, Step, Task, Computer).join(Step, Step.id == Log.step, isouter=True). \
            join(Task, Task.id == Step.task, isouter=True). \
            join(Computer, Computer.name == Task.computer_assigned, isouter=True)
        if filter.get('dag'):
            query = query.filter(Task.dag == filter['dag'])
        if filter.get('task'):
            query = query.filter(Task.id == filter['task'])

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
        for log, step, task, computer in self.paginator(query, options):
            item = {
                'id': log.id,
                'message': log.message,
                'time': self.serializer.serialize_date(log.time),
                'level': log_name(log.level),
                'component': to_snake(ComponentType(log.component).name),
                'computer': computer.to_dict() if computer else None,
                'step': step.to_dict() if step else None,
                'task': task.to_dict() if task else None
            }
            data.append(item)

        return {'total': total, 'data': data}
