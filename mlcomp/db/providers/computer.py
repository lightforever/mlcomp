import json
from collections import defaultdict

from mlcomp.db.providers.base import *
import datetime


class ComputerProvider(BaseDataProvider):
    model = Computer

    def computers(self):
        return {c.name: {k: v for k, v in c.__dict__.items()} for c in self.query(Computer).all()}

    def get(self, filter: dict, options:PaginatorOptions=None):
        query = self.query(Computer)
        total = query.count()
        if options:
            query = self.paginator(query, options)
        res = []
        for c in query.all():
            item = self.to_dict(c)
            item['usage'] = json.loads(item['usage']) if item['usage'] else {'cpu': 0, 'memory': 0,
                                                        'gpu': [{'memory': 0, 'load': 0} for i in range(item['gpu'])]}
            item['memory'] = int(item['memory']/1000)
            item['usage']['cpu'] = int(item['usage']['cpu'])
            item['usage']['memory'] = int(item['usage']['memory'])
            for gpu in item['usage']['gpu']:
                gpu['memory'] = int(gpu['memory']*100)
                gpu['load'] = int(gpu['load']*100)

            item['usage_history'] = self.usage_history(c.name, filter.get('usage_min_time'))
            res.append(item)

        return {'data': res, 'total': total}

    def usage_history(self, computer: str, min_time: datetime=None):
        min_time = min_time or (now() - datetime.timedelta(days=1))
        query = self.query(ComputerUsage).filter(ComputerUsage.time >= min_time).filter(
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
        computer = self.query(Computer).filter(Computer.name==name).first()
        computer.usage = json.dumps(usage)
        self.update()


