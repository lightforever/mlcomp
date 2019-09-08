from collections import OrderedDict

from mlcomp.db.report_info.item import ReportLayoutItem


class ReportLayoutImgSegment(ReportLayoutItem):
    def __init__(
        self,
        name: str
    ):
        super().__init__(name)

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')

        assert len(value) == 0, f'Unknown parameter in ' \
            f'report.img_segment={value.popitem()}'
        return cls(
            name
        )


__all__ = ['ReportLayoutImgSegment']
