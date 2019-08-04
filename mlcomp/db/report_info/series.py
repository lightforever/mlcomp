from collections import OrderedDict

from mlcomp.db.report_info.item import ReportLayoutItem


class ReportLayoutSeries(ReportLayoutItem):
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


__all__ = ['ReportLayoutSeries']
