from typing import List

from catalyst.dl.callbacks.core import Callback
from catalyst.dl.state import RunnerState

from db.providers import ReportSeriesProvider, ReportImgProvider
from db.models import ReportSeries, ReportImg
from utils.config import Config
from .base import Executor
from pathlib import Path

from catalyst.utils.config import parse_args_uargs, dump_config
from catalyst.utils.misc import set_global_seeds
from catalyst.dl.scripts.utils import import_experiment_and_runner, dump_code
from catalyst.dl.experiments.runner import Runner


class Args:
    baselogdir = None
    batch_size = None
    check = False
    config = None
    expdir = None
    logdir = None
    num_epochs = None
    num_workers = None
    resume = None
    seed = 42
    verbose = False

    def _get_kwargs(self):
        return [(k, v) for k, v in self.__dict__.items() if not k.startswith('_')]


@Executor.register
class Catalyst(Executor, Callback):
    def __init__(self, args: Args, series: List[str]):
        self.args = args
        self.series_provider = ReportSeriesProvider()
        self.img_provider = ReportImgProvider()
        self.series = series

    def on_epoch_end(self, state: RunnerState):
        for s in self.series:
            train = state.metrics.epoch_values['train'][s]
            val = state.metrics.epoch_values['valid'][s]

            train = ReportSeries(group='train', name=s, epoch=state.epoch, task=self.task.id, value=train)
            val = ReportSeries(group='valid', name=s, epoch=state.epoch, task=self.task.id, value=val)

            self.series_provider.add(train)
            self.series_provider.add(val)

    def on_stage_start(self, state: RunnerState):
        state.loggers = []

    @classmethod
    def _from_config(cls, executor: dict, config: Config):
        args = Args()
        for k, v in executor['args'].items():
            if v in ['False', 'True']:
                v = v == 'True'
            elif v.isnumeric():
                v = int(v)

            setattr(args, k, v)
        return cls(args=args, series=executor['series'])

    def work(self):
        args, config = parse_args_uargs(self.args, [])
        set_global_seeds(config.get("seed", 42))

        Experiment, R = import_experiment_and_runner(Path(args.expdir))

        experiment = Experiment(config)
        runner: Runner = R()

        _get_callbacks = experiment.get_callbacks

        def get_callbacks(stage):
            return _get_callbacks(stage) + [self]

        experiment.get_callbacks = get_callbacks

        runner.run_experiment(
            experiment,
            check=args.check
        )
