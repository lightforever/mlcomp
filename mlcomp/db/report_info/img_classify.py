from collections import OrderedDict

from mlcomp.db.report_info.item import ReportLayoutItem


class ReportLayoutImgClassify(ReportLayoutItem):
    def __init__(
        self,
        name: str,
        confusion_matrix: bool,
    ):
        super().__init__(name)

        self.confusion_matrix = confusion_matrix

    @classmethod
    def from_dict(cls, name: str, value: OrderedDict):
        value.pop('type')
        confusion_matrix = value.pop('confusion_matrix', False)

        assert len(value) == 0, f'Unknown parameter in ' \
            f'report.img_classify={value.popitem()}'
        return cls(
            name,
            confusion_matrix=confusion_matrix
        )


__all__ = ['ReportLayoutImgClassify']
