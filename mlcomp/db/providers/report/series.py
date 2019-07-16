from mlcomp.db.models import ReportSeries
from mlcomp.db.providers import BaseDataProvider


class ReportSeriesProvider(BaseDataProvider):
    model = ReportSeries
