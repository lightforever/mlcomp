from pathlib import Path

from catalyst.dl import RunnerState, Callback, Runner
from catalyst.dl.callbacks import VerboseLogger
from catalyst.dl.utils.scripts import import_experiment_and_runner
from catalyst.utils.config import parse_args_uargs, dump_config

from mlcomp.db.providers import TaskProvider, ReportSeriesProvider
from mlcomp.utils.misc import now
from mlcomp.db.models import ReportSeries
from mlcomp.utils.config import Config
from mlcomp.worker.executors.base import Executor
from mlcomp.db.report_info import *
from mlcomp.worker.executors.catalyst.precision_recall import \
    PrecisionRecallCallback
from mlcomp.worker.executors.catalyst.f1 import F1Callback
from mlcomp.worker.executors.catalyst.img_classify import ImgClassifyCallback


class Args:
    baselogdir = None
    batch_size = None
    check = False
    config = None
    configs = []
    expdir = None
    logdir = None
    num_epochs = None
    num_workers = None
    resume = None
    seed = 42
    verbose = True

    def _get_kwargs(self):
        return [(k, v) for k, v in self.__dict__.items() if
                not k.startswith('_')]


# noinspection PyTypeChecker
@Executor.register
class Catalyst(Executor, Callback):
    __syn__ = 'catalyst'

    def __init__(self, args: Args, report: ReportLayoutInfo):
        self.args = args
        self.report = report
        self.experiment = None
        self.runner = None
        self.series_provider = ReportSeriesProvider()
        self.task_provider = TaskProvider()

    def callbacks(self):
        result = [self]
        for items, cls in [
            [self.report.precision_recall, PrecisionRecallCallback],
            [self.report.f1, F1Callback],
            [self.report.img_classify, ImgClassifyCallback],
        ]:
            for item in items:
                result.append(cls(self.experiment, self.task, self.dag, item))
        return result

    def on_epoch_end(self, state: RunnerState):
        for s in self.report.series:
            train = state.metrics.epoch_values['train'][s.key]
            val = state.metrics.epoch_values['valid'][s.key]

            train = ReportSeries(part='train',
                                 name=s.key,
                                 epoch=state.epoch,
                                 task=self.task.id,
                                 value=train,
                                 time=now())

            val = ReportSeries(part='valid',
                               name=s.key,
                               epoch=state.epoch,
                               task=self.task.id,
                               value=val,
                               time=now())

            self.series_provider.add(train)
            self.series_provider.add(val)

            if s.key == self.report.metric.name:
                best = False
                if self.report.metric.minimize:
                    if self.task.score is None or val.value < self.task.score:
                        best = True
                else:
                    if self.task.score is None or val.value > self.task.score:
                        best = True
                if best:
                    self.task.score = val.value
                    self.task_provider.update()

    def on_stage_start(self, state: RunnerState):
        state.loggers = [VerboseLogger()]

    @classmethod
    def _from_config(cls,
                     executor: dict,
                     config: Config,
                     additional_info: dict):
        args = Args()
        for k, v in executor['args'].items():
            if v in ['False', 'True']:
                v = v == 'True'
            elif v.isnumeric():
                v = int(v)

            setattr(args, k, v)

        report = ReportLayoutInfo(additional_info.get('report_config', dict()))
        if len(args.configs) == 0:
            args.configs = [args.config]
        return cls(args=args, report=report)

    def work(self):
        args, config = parse_args_uargs(self.args, [])
        # set_global_seed(args.seed)

        Experiment, R = import_experiment_and_runner(Path(args.expdir))

        experiment = Experiment(config)
        runner: Runner = R()

        self.experiment = experiment
        self.runner = runner

        _get_callbacks = experiment.get_callbacks

        def get_callbacks(stage):
            return _get_callbacks(stage) + self.callbacks()

        experiment.get_callbacks = get_callbacks

        if experiment.logdir is not None:
            dump_config(config, experiment.logdir, args.configs)

        runner.run_experiment(
            experiment,
            check=args.check
        )
