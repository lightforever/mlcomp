from collections import OrderedDict
from typing import List
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, classification_report

from mlcomp.utils.plot import figure_to_binary, plot_classification_report


class ReportSchemeItem:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        assert len(value) == 0, f'Unknown parameter in ' \
            f'report info item = {name}: {value.popitem()}'
        return cls(name)


class ReportSchemeSeries(ReportSchemeItem):
    def __init__(self, name: str, key: str):
        super().__init__(name)

        self.key = key

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        assert 'key' in value, f'report.series={name}. key is required'
        value.pop('type')
        key = value.pop('key')

        assert len(value) == 0, f'Unknown parameter in ' \
            f'report.series={name}: {value.popitem()}'
        return cls(name, key)


class ReportSchemePrecisionRecall(ReportSchemeItem):
    def plot(self, y: np.array, pred: np.array):
        p, r, t = precision_recall_curve(y, pred)
        fig, ax = plt.subplots(figsize=(4.2, 2.7))
        ax2 = ax.twinx()

        t = np.hstack([t, t[-1]])

        ax.plot(r, p)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax2.set_ylabel('Threashold')
        ax2.plot(r, t, c='red')
        return figure_to_binary(fig)


class ReportSchemeF1(ReportSchemeItem):
    def plot(self, y: np.array, pred: np.array):
        report = classification_report(y, pred)
        fig = plot_classification_report(report)
        return figure_to_binary(fig, dpi=70)


class ReportSchemeMetric:
    def __init__(self, name: str, minimize: bool):
        self.name = name
        self.minimize = minimize

    @staticmethod
    def from_dict(data: dict):
        name = data.pop('name')
        minimize = data.pop('minimize')
        assert len(data) == 0, f'Unknown parameter in ' \
            f'report.metric={data.popitem()}'
        return ReportSchemeMetric(name, minimize)


class ReportSchemeImgClassify(ReportSchemeItem):
    def __init__(self,
                 name: str,
                 epoch_every: int,
                 count_class_max: int,
                 train: bool,
                 threshold=None):
        super().__init__(name)

        self.epoch_every = epoch_every
        self.count_class_max = count_class_max
        self.train = train
        self.threshold = threshold

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        epoch_every = value.pop('epoch_every', None)
        count_class_max = value.pop('count_class_max', None)
        train = value.pop('train', False)
        threshold = value.pop('threshold', dict())

        assert len(value) == 0, f'Unknown parameter in ' \
            f'report.img_classify={value.popitem()}'
        return cls(name,
                   epoch_every=epoch_every,
                   count_class_max=count_class_max,
                   train=train, threshold=threshold)


class ReportSchemeInfo:
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

    def _get_img_classify(self) -> List[ReportSchemeImgClassify]:
        return self._by_type('img_classify', ReportSchemeImgClassify)

    def _get_f1(self) -> List[ReportSchemeF1]:
        return self._by_type('f1', ReportSchemeF1)

    def _get_series(self) -> List[ReportSchemeSeries]:
        return self._by_type('series', ReportSchemeSeries)

    def _get_precision_recall(self) -> List[ReportSchemePrecisionRecall]:
        return self._by_type('precision_recall', ReportSchemePrecisionRecall)

    def _get_metric(self) -> ReportSchemeMetric:
        return ReportSchemeMetric.from_dict(self.data['metric'])

    @classmethod
    def union_schemes(cls, name: str, schemes: dict, return_dict: bool = True):
        assert name in schemes, f'Scheme {name} is not in the collection'
        l = deepcopy(schemes[name])
        r = dict()
        if l.get('extend'):
            assert l['extend'] in schemes, \
                f'Scheme for extending = {l["extend"]}' \
                    f' is not in the collection'
            r = cls.union_schemes(l['extend'], schemes)

        if 'metric' in l:
            r['metric'] = l['metric']

        if 'items' in l:
            r['items'] = r.get('items', dict())
            r['items'].update(l['items'])

        if 'layout' in l:
            r['layout'] = r.get('layout', []) + l['layout']

        if return_dict:
            return r

        return ReportSchemeInfo(r)


__all__ = [
    'ReportSchemeItem',
    'ReportSchemeSeries',
    'ReportSchemePrecisionRecall',
    'ReportSchemeF1',
    'ReportSchemeMetric',
    'ReportSchemeImgClassify',
    'ReportSchemeInfo']
