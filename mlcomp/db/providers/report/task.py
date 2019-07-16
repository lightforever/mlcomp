from mlcomp.db.models import ReportTasks
from mlcomp.db.providers import BaseDataProvider


class ReportTasksProvider(BaseDataProvider):
    model = ReportTasks


__all__ = ['ReportTasksProvider']
