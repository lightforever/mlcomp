import json
import datetime
from collections import defaultdict

from sqlalchemy import func

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.enums import TaskStatus
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.db.models import Computer, ComputerUsage, Task, Docker
from mlcomp.utils.misc import now


class ComputerProvider(BaseDataProvider):
    model = Computer

    def computers(self):
        return {c.name: {k: v for k, v in c.__dict__.items()}
                for c in self.query(Computer).all()}

    def get(self, filter: dict, options: PaginatorOptions = None):
        query = self.query(Computer)
        total = query.count()
        if options:
            query = self.paginator(query, options)
        res = []
        for c in query.all():
            item = self.to_dict(c)
            default_usage = {
                'cpu': 0,
                'memory': 0,
                'gpu': [{'memory': 0, 'load': 0} for i in range(item['gpu'])]
            }
            item['usage'] = json.loads(item['usage']) \
                if item['usage'] else default_usage
            item['memory'] = int(item['memory'] / 1000)
            item['usage']['cpu'] = int(item['usage']['cpu'])
            item['usage']['memory'] = int(item['usage']['memory'])
            for gpu in item['usage']['gpu']:
                gpu['memory'] = int(gpu['memory'] * 100)
                gpu['load'] = int(gpu['load'] * 100)

            item['usage_history'] = self.usage_history(
                c.name,
                filter.get('usage_min_time'))
            item['dockers'] = self.dockers(c.name, c.cpu)
            res.append(item)

        return {'data': res, 'total': total}

    def usage_history(self, computer: str, min_time: datetime = None):
        min_time = min_time or (now() - datetime.timedelta(days=1))
        query = self.query(ComputerUsage).filter(
            ComputerUsage.time >= min_time).filter(
            ComputerUsage.computer == computer).order_by(ComputerUsage.time)
        res = {'time': [], 'mean': []}
        mean = defaultdict(list)
        for c in query.all():
            item = self.to_dict(c, datetime_format='%Y-%m-%d %H:%M:%SZ')
            usage = json.loads(item['usage'])
            res['time'].append(item['time'])

            mean['cpu'].append(usage['mean']['cpu'])
            mean['memory'].append(usage['mean']['memory'])
            for i, gpu in enumerate(usage['mean']['gpu']):
                mean[f'gpu_{i}'].append(gpu['load'])

        for item in mean:
            res['mean'].append({'name': item, 'value': mean[item]})

        return dict(res)

    def current_usage(self, name: str, usage: dict):
        computer = self.query(Computer).filter(Computer.name == name).first()
        computer.usage = json.dumps(usage)
        self.update()

    def by_name(self, name: str):
        return self.query(Computer).filter(Computer.name == name).one()

    def computers_have_succeeded_tasks(self, min_time: datetime):
        res = self.session.query(Task.computer_assigned). \
            filter(Task.finished >= min_time). \
            filter(Task.status == TaskStatus.Success.value). \
            all()
        return [r[0] for r in res]

    def dockers(self, computer: str, cpu: int):
        res = self.query(Docker, func.count(Task.computer).filter(
            Task.status == TaskStatus.InProgress.value).label('count')). \
            join(Task, Task.computer_assigned == computer, isouter=True). \
            filter(Docker.computer == computer). \
            group_by(Docker.name, Docker.computer). \
            all()
        return [{
            'name': r[0].name,
            'last_activity': self.serialize_datetime_long(r[0].last_activity),
            'in_progress': r[1],
            'free': cpu - r[1]}
            for r in res]


__all__ = ['ComputerProvider']
