from mlcomp.db.core import PaginatorOptions
from mlcomp.db.models import Memory
from mlcomp.db.providers.base import BaseDataProvider


class MemoryProvider(BaseDataProvider):
    model = Memory

    def get(self, filter: dict, options: PaginatorOptions = None):
        query = self.query(Memory)
        if filter.get('model'):
            query = query.filter(Memory.model.contains(filter['model']))
        if filter.get('variant'):
            query = query.filter(Memory.variant.contains(filter['variant']))

        total = query.count()
        paginator = self.paginator(query, options) if options else query
        data = []
        for p in paginator.all():
            item = self.to_dict(p)
            data.append(item)

        return {
            'total': total,
            'data': data
        }

    def find(self, data: dict):
        query = self.query(Memory)
        for k, v in data.items():
            if k in ['batch_size']:
                continue

            if hasattr(Memory, k):
                query = query.filter(getattr(Memory, k) == v)
        return query.all()


__all__ = ['MemoryProvider']
