from mlcomp.db.models import ReportSeries
from mlcomp.db.providers.base import BaseDataProvider


class ReportSeriesProvider(BaseDataProvider):
    model = ReportSeries


__all__ = ['ReportSeriesProvider']
