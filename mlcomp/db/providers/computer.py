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
        total =query.count()
        if options:
            query = self.paginator(query, options)
        res = []
        for c in query.all():
            item = c.to_dict()
            item['usage'] = json.loads(item['usage'])
            item['memory'] = int(item['memory']/1000)
            item['usage']['cpu'] = int(item['usage']['cpu'])
            item['usage']['memory'] = int(item['usage']['memory'])
            for gpu in item['usage']['gpu']:
                gpu['memory'] = int(gpu['memory']*100)
                gpu['load'] = int(gpu['load']*100)

            item['usage_history'] = self.usage_history(c.name)
            res.append(item)

        return {'data': res, 'total': total}

    def usage_history(self, computer: str, min_time: datetime=None):
        min_time = min_time or (now() - datetime.timedelta(days=1))
        query = self.query(ComputerUsage).filter(ComputerUsage.time >= min_time).filter(
            ComputerUsage.computer == computer).order_by(ComputerUsage.time)
        res = {'time': [], 'mean': [], 'peak': []}
        mean = defaultdict(list)
        peak = defaultdict(list)
        for c in query.all():
            item = c.to_dict()
            usage = json.loads(item['usage'])
            res['time'].append(item['time'])
            
            mean['cpu'].append(usage['mean']['cpu'])
            mean['memory'].append(usage['mean']['memory'])
            for i, gpu in enumerate(usage['mean']['gpu']):
                mean[f'gpu_{i}'].append(gpu['load'])

            peak['cpu'].append(usage['peak']['cpu'])
            peak['memory'].append(usage['peak']['memory'])
            for i, gpu in enumerate(usage['peak']['gpu']):
                peak[f'gpu_{i}'].append(gpu['load'])

        for item in mean:
            res['mean'].append({'name': item, 'value': mean[item]})

        for item in peak:
            res['peak'].append({'name': item, 'value': peak[item]})

        return dict(res)

    def current_usage(self, name: str, usage: dict):
        computer = self.query(Computer).filter(Computer.name==name).first()
        computer.usage = json.dumps(usage)
        self.update()


