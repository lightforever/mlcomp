import os
from pathlib import Path

from catalyst.dl import RunnerState, Callback, Runner
from catalyst.dl.callbacks import VerboseLogger
from catalyst.dl.utils.scripts import import_experiment_and_runner
from catalyst.utils.config import parse_args_uargs, dump_config

from mlcomp.contrib.search.grid import grid_cells
from mlcomp.db.providers import TaskProvider, ReportSeriesProvider
from mlcomp.utils.misc import now
from mlcomp.db.models import ReportSeries
from mlcomp.utils.config import Config, merge_dicts_smart
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

    def __init__(self, args: Args,
                 report: ReportLayoutInfo,
                 distr_info: dict,
                 grid_config: dict):

        self.distr_info = distr_info
        self.args = args
        self.report = report
        self.experiment = None
        self.runner = None
        self.series_provider = ReportSeriesProvider()
        self.task_provider = TaskProvider()
        self.grid_config = grid_config
        self.master = True

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

        report_config = additional_info.get('report_config', dict())
        grid_cell = additional_info.get('grid_cell')
        report = ReportLayoutInfo(report_config)
        if len(args.configs) == 0:
            args.configs = [args.config]

        grid_config = {}
        if grid_cell is not None:
            grid_config = grid_cells(executor['grid'])[grid_cell][0]
        distr_info = {}
        if 'distr_info' in additional_info:
            distr_info = additional_info['distr_info']

        return cls(args=args,
                   report=report,
                   grid_config=grid_config,
                   distr_info=distr_info
                   )

    def set_dist_env(self, config):
        info = self.distr_info
        os.environ['MASTER_ADDR'] = info['master_addr']
        os.environ['MASTER_PORT'] = str(info['master_port'])
        os.environ['WORLD_SIZE'] = str(self.task.gpu)

        os.environ['RANK'] = str(info['rank'])
        os.environ['LOCAL_RANK'] = str(info['local_rank'])
        os.environ['CUDA_VISIBLE_DEVICES'] =  \
            ','.join([str(i) for i in info['gpu_visible']])

        config['distributed_params'] = {'rank': info['rank']}

        if info['rank'] > 0:
            self.master = False

    def parse_args_uargs(self):
        args, config = parse_args_uargs(self.args, [])
        config = merge_dicts_smart(config, self.grid_config)
        if self.distr_info:
            self.set_dist_env(config)
        return args, config

    def work(self):
        args, config = self.parse_args_uargs()
        # set_global_seed(args.seed)

        Experiment, R = import_experiment_and_runner(Path(args.expdir))

        experiment = Experiment(config)
        runner: Runner = R()

        self.experiment = experiment
        self.runner = runner

        _get_callbacks = experiment.get_callbacks

        def get_callbacks(stage):
            return _get_callbacks(stage) + \
                   self.callbacks() if self.master else []

        experiment.get_callbacks = get_callbacks

        if experiment.logdir is not None:
            dump_config(config, experiment.logdir, args.configs)

        runner.run_experiment(
            experiment,
            check=args.check
        )