from collections import OrderedDict
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mlcomp.utils.plot import figure_to_binary

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
        multi = value.pop('multi') if 'multi' in value else None
        single_group = value.pop('single_group') if 'single_group' in value else None

        assert len(value) == 0, f'Unknown parameter in report.series={name}: {value.popitem()}'
        return cls(name, key, multi=multi, single_group=single_group)


class ReportInfoPrecisionRecall(ReportInfoItem):
    def plot(self, p: np.array, r: np.array, t: np.array):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
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


class ReportInfo:
    def __init__(self, name: str, data: OrderedDict):
        self.name = name
        self.data = data
        self.series = self._get_series()
        self.precision_recall = self._get_precision_recall()

    def has_classification(self):
        return len(self.precision_recall) > 0

    def _get_series(self) -> List[ReportInfoSeries]:
        return [ReportInfoSeries.from_dict(k, v) for k, v in self.data.items() if v.get('type') == 'series']

    def _get_precision_recall(self) -> List[ReportInfoPrecisionRecall]:
        return [ReportInfoPrecisionRecall.from_dict(k, v) for k, v in self.data.items() if
                v.get('type') == 'precision_recall']
