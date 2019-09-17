from collections import OrderedDict

from mlcomp.db.report_info.item import ReportLayoutItem


class ReportLayoutImgSegment(ReportLayoutItem):
    def __init__(
        self,
        name: str,
        max_height: int,
        max_width: int
    ):
        super().__init__(name)

        self.max_height = max_height
        self.max_width = max_width

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        max_height = value.pop('max_height', None)
        max_width = value.pop('max_width', None)

        assert len(value) == 0, f'Unknown parameter in ' \
            f'report.img_segment={value.popitem()}'

        return cls(
            name,
            max_height=max_height,
            max_width=max_width
        )


__all__ = ['ReportLayoutImgSegment']
