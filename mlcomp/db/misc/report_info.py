from collections import OrderedDict
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mlcomp.utils.plot import figure_to_binary, plot_classification_report
from sklearn.metrics import precision_recall_curve, classification_report


class ReportInfoItem:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        assert len(value) == 0, f'Unknown parameter in report info item = {name}: {value.popitem()}'
        return cls(name)


class ReportInfoSeries(ReportInfoItem):
    def __init__(self, name: str, key: str, multi: str = None, single_group: str = None):
        super(ReportInfoSeries, self).__init__(name)

        self.key = key
        self.multi = multi
        self.single_group = single_group

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        assert 'key' in value, f'report.series={name}. key is required'
        value.pop('type')
        key = value.pop('key')
        multi = value.pop('multi', 'multi')
        single_group = value.pop('single_group', 'single_group')

        assert len(value) == 0, f'Unknown parameter in report.series={name}: {value.popitem()}'
        return cls(name, key, multi=multi, single_group=single_group)


class ReportInfoPrecisionRecall(ReportInfoItem):
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


class ReportInfoF1(ReportInfoItem):
    def plot(self, y: np.array, pred: np.array):
        report = classification_report(y, pred)
        fig = plot_classification_report(report)
        return figure_to_binary(fig, dpi=70)


class ReportInfoMetric:
    def __init__(self, name: str, minimize: bool):
        self.name = name
        self.minimize = minimize

    @staticmethod
    def from_dict(data: dict):
        name = data.pop('name')
        minimize = data.pop('minimize')
        assert len(data) == 0, f'Unknown parameter in report.metric={data.popitem()}'
        return ReportInfoMetric(name, minimize)


class ReportInfoImgClassify(ReportInfoItem):
    def __init__(self, name: str,  epoch_every: int, count_class_max: int, train: bool):
        super(ReportInfoImgClassify, self).__init__(name)
        self.epoch_every = epoch_every
        self.count_class_max = count_class_max
        self.train = train

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        epoch_every = value.pop('epoch_every', None)
        count_class_max = value.pop('count_class_max', None)
        train = value.pop('train', False)

        assert len(value) == 0, f'Unknown parameter in report.img_classify={value.popitem()}'
        return cls(name, epoch_every=epoch_every, count_class_max=count_class_max, train=train)


class ReportInfo:
    def __init__(self, data: OrderedDict):
        self.data = data
        self.series = self._get_series()
        self.precision_recall = self._get_precision_recall()
        self.metric = self._get_metric()
        self.f1 = self._get_f1()
        self.img_classify = self._get_img_classify()

    def has_classification(self):
        return len(self.precision_recall) > 0

    def _get_img_classify(self) -> List[ReportInfoImgClassify]:
        return [ReportInfoImgClassify.from_dict(k, v) for k, v in self.data['items'].items()
                if v.get('type') == 'img_classify']

    def _get_f1(self) -> List[ReportInfoF1]:
        return [ReportInfoF1.from_dict(k, v) for k, v in self.data['items'].items() if v.get('type') == 'f1']

    def _get_series(self) -> List[ReportInfoSeries]:
        return [ReportInfoSeries.from_dict(k, v) for k, v in self.data['items'].items() if v.get('type') == 'series']

    def _get_precision_recall(self) -> List[ReportInfoPrecisionRecall]:
        return [ReportInfoPrecisionRecall.from_dict(k, v) for k, v in self.data['items'].items() if
                v.get('type') == 'precision_recall']

    def _get_metric(self) -> ReportInfoMetric:
        return ReportInfoMetric.from_dict(self.data['metric'])


__all__ = ['ReportInfoItem', 'ReportInfoSeries', 'ReportInfoPrecisionRecall',
           'ReportInfoF1', 'ReportInfoMetric', 'ReportInfoImgClassify', 'ReportInfo']
