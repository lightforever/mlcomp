import json
import datetime
from collections import defaultdict

from sqlalchemy import func, case

from mlcomp.db.core import PaginatorOptions
from mlcomp.db.enums import TaskStatus
from mlcomp.db.providers.base import BaseDataProvider
from mlcomp.db.models import Computer, ComputerUsage, Task, Docker
from mlcomp.utils.misc import now, parse_time


class ComputerProvider(BaseDataProvider):
    model = Computer

    def computers(self):
        return {
            c.name:
            {k: v
             for k, v in c.__dict__.items() if not k.startswith('_')}
            for c in self.query(Computer).all()
        }

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
                'gpu': [{
                    'memory': 0,
                    'load': 0
                } for i in range(item['gpu'])]
            }
            sync_status = 'Not synced'
            sync_date = None
            if c.last_synced:
                sync_date = self.serialize_datetime(c.last_synced)
                sync_status = f'Last synced'

            if c.syncing_computer:
                if c.last_synced is None or \
                        (now() - c.last_synced).total_seconds() >= 5:
                    sync_status = f'Syncing with {c.syncing_computer}'
                    if c.last_synced:
                        sync_status += f' from '
                    sync_date = self.serialize_datetime(c.last_synced)

            item['sync_status'] = sync_status
            item['sync_date'] = sync_date

            item['usage'] = json.loads(item['usage']) \
                if item['usage'] else default_usage
            item['memory'] = int(item['memory'] / 1000)
            item['usage']['cpu'] = int(item['usage']['cpu'])
            item['usage']['memory'] = int(item['usage']['memory'])
            for gpu in item['usage']['gpu']:
                gpu['memory'] = int(gpu['memory'])
                gpu['load'] = int(gpu['load'])

            min_time = parse_time(filter.get('usage_min_time'))

            item['usage_history'] = self.usage_history(
                c.name, min_time
            )
            item['dockers'] = self.dockers(c.name, c.cpu)
            res.append(item)

        return {'data': res, 'total': total}

    def usage_history(self, computer: str, min_time: datetime = None):
        min_time = min_time or (now() - datetime.timedelta(days=1))
        query = self.query(ComputerUsage).filter(
            ComputerUsage.time >= min_time
        ).filter(ComputerUsage.computer == computer
                 ).order_by(ComputerUsage.time)
        res = {'time': [], 'mean': []}
        mean = defaultdict(list)
        for c in query.all():
            item = self.to_dict(c, datetime_format='%Y-%m-%d %H:%M:%SZ')
            usage = json.loads(item['usage'])
            res['time'].append(item['time'])

            mean['cpu'].append(usage['mean']['cpu'])
            mean['memory'].append(usage['mean']['memory'])
            mean['disk'].append(usage['mean']['disk'])
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
        res = self.session.query(Task.computer_assigned.distinct()). \
            filter(Task.finished >= min_time). \
            filter(Task.status == TaskStatus.Success.value). \
            all()
        res = [r[0] for r in res]
        return self.session.query(Computer). \
            filter(Computer.name.in_(res)). \
            all()

    def dockers(self, computer: str, cpu: int):
        count_cond = func.sum(
            case(
                whens=[(Task.status == TaskStatus.InProgress.value, 1)],
                else_=0
            ).label('count')
        )

        res = self.query(Docker, count_cond). \
            join(Task, Task.computer_assigned == computer, isouter=True). \
            filter(Docker.computer == computer). \
            group_by(Docker.name, Docker.computer). \
            all()
        return [
            {
                'name': r[0].name,
                'last_activity': self.serialize_datetime(
                    r[0].last_activity
                ),
                'in_progress': r[1],
                'free': cpu - r[1]
            } for r in res
        ]

    def all_with_last_activtiy(self):
        query = self.query(Computer, func.max(Docker.last_activity)). \
            join(Docker, Docker.computer == Computer.name). \
            group_by(Computer.name)
        res = []
        for c, a in query.all():
            c.last_activity = a
            res.append(c)
        return res


__all__ = ['ComputerProvider']
