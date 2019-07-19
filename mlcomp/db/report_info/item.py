from collections import OrderedDict


class ReportLayoutItem:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        assert len(value) == 0, f'Unknown parameter in ' \
            f'report info item = {name}: {value.popitem()}'
        return cls(name)


__all__ = ['ReportLayoutItem']