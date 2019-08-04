from collections import defaultdict

from catalyst.dl import Callback, Experiment, RunnerState

from mlcomp.db.providers import ReportImgProvider
from mlcomp.db.report_info import ReportLayoutItem
from mlcomp.db.models import Task, Dag


class BaseCallback(Callback):
    def __init__(
        self,
        experiment: Experiment,
        task: Task,
        dag: Dag,
        info: ReportLayoutItem
    ):
        self.info = info
        self.task = task
        self.dag = dag
        self.img_provider = ReportImgProvider()
        self.experiment = experiment
        self.is_best = False
        self.data = defaultdict(lambda: defaultdict(list))
        self.added = defaultdict(lambda: defaultdict(int))

    def on_epoch_end(self, state: RunnerState):
        self.is_best = state.metrics.is_best
        self.data = defaultdict(lambda: defaultdict(list))
        self.added = defaultdict(lambda: defaultdict(int))
