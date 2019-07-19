from collections import OrderedDict
from typing import List
from copy import deepcopy

from mlcomp.db.report_info.f1 import ReportLayoutF1
from mlcomp.db.report_info.img_classify import ReportLayoutImgClassify
from mlcomp.db.report_info.metric import ReportLayoutMetric
from mlcomp.db.report_info.precision_recall import ReportLayoutPrecisionRecall
from mlcomp.db.report_info.series import ReportLayoutSeries


class ReportLayoutInfo:
    def __init__(self, data: OrderedDict):
        assert 'items' in data, 'no items in report'
        assert 'metric' in data, 'no metric in report'

        self.data = data
        self.series = self._get_series()
        self.precision_recall = self._get_precision_recall()
        self.metric = self._get_metric()
        self.f1 = self._get_f1()
        self.img_classify = self._get_img_classify()
        self.layout = {'type': 'root', 'items': data.get('layout', [])}
        self._check_layout(self.layout)

    def _check_layout(self, item):
        types = ['root', 'panel', 'blank', 'series', 'img_classify',
                 'img']
        assert item.get('type') in types, f'Unknown item type = {item["type"]}'

        fields = {
            'root': [('items', False)],
            'panel': [
                'title',
                ('parent_cols', False),
                ('cols', False),
                ('row_height', False),
                ('rows', False),
                ('items', False),
                ('expanded', False),
                ('table', False)
            ],
            'blank': [
                ('cols', False),
                ('rows', False)
            ],
            'series': [
                ('multi', False),
                ('group', False),
                'source',
                ('cols', False),
                ('rows', False)],
            **{k: [
                'source',
                ('cols', False),
                ('rows', False)]
                for k in types[4:]}
        }
        keys = set(item.keys()) - {'type'}
        for f in fields[item['type']]:
            req = True
            if type(f) == tuple:
                f, req = f
            if req and f not in item:
                raise Exception(f'Type {item["type"]} must contain field {f}')
            if f in keys:
                keys.remove(f)
        assert len(keys) == 0, \
            f'Unknown fields {keys} for type = {item["type"]}'

        if 'items' in item:
            for item in item['items']:
                self._check_layout(item)

    def has_classification(self):
        return len(self.precision_recall) > 0

    def _by_type(self, t: str, c):
        return [c.from_dict(k, v)
                for k, v in self.data['items'].items() if v.get('type') == t]

    def _get_img_classify(self) -> List[ReportLayoutImgClassify]:
        return self._by_type('img_classify', ReportLayoutImgClassify)

    def _get_f1(self) -> List[ReportLayoutF1]:
        return self._by_type('f1', ReportLayoutF1)

    def _get_series(self) -> List[ReportLayoutSeries]:
        return self._by_type('series', ReportLayoutSeries)

    def _get_precision_recall(self) -> List[ReportLayoutPrecisionRecall]:
        return self._by_type('precision_recall', ReportLayoutPrecisionRecall)

    def _get_metric(self) -> ReportLayoutMetric:
        return ReportLayoutMetric.from_dict(self.data['metric'])

    @classmethod
    def union_layouts(cls, name: str, layouts: dict, return_dict: bool = True):
        assert name in layouts, f'Layout {name} is not in the collection'
        l = deepcopy(layouts[name])
        r = dict()
        if l.get('extend'):
            assert l['extend'] in layouts, \
                f'Layout for extending = {l["extend"]}' \
                    f' is not in the collection'
            r = cls.union_layouts(l['extend'], layouts)

        if 'metric' in l:
            r['metric'] = l['metric']

        if 'items' in l:
            r['items'] = r.get('items', dict())
            r['items'].update(l['items'])

        if 'layout' in l:
            r['layout'] = r.get('layout', []) + l['layout']

        if return_dict:
            return r

        return ReportLayoutInfo(r)


__all__ = ['ReportLayoutInfo']