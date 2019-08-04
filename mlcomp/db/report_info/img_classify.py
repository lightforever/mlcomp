from collections import OrderedDict

from mlcomp.db.report_info.item import ReportLayoutItem


class ReportLayoutImgClassify(ReportLayoutItem):
    def __init__(
        self,
        name: str,
        epoch_every: int,
        count_class_max: int,
        train: bool,
        threshold=None
    ):
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
        return cls(
            name,
            epoch_every=epoch_every,
            count_class_max=count_class_max,
            train=train,
            threshold=threshold
        )


__all__ = ['ReportLayoutImgClassify']
