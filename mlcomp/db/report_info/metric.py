class ReportLayoutMetric:
    def __init__(self, name: str, minimize: bool):
        self.name = name
        self.minimize = minimize

    @staticmethod
    def from_dict(data: dict):
        name = data.pop('name')
        minimize = data.pop('minimize')
        assert len(data) == 0, f'Unknown parameter in ' \
            f'report.metric={data.popitem()}'
        return ReportLayoutMetric(name, minimize)


__all__ = ['ReportLayoutMetric']