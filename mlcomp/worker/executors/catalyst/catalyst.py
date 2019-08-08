import os
from pathlib import Path

from catalyst.utils import set_global_seed
from catalyst.dl import RunnerState, Callback, Runner, CheckpointCallback
from catalyst.dl.callbacks import VerboseLogger, RaiseExceptionLogger
from catalyst.dl.utils.scripts import import_experiment_and_runner
from catalyst.utils.config import parse_args_uargs, dump_config

from mlcomp.contrib.search.grid import grid_cells
from mlcomp.db.providers import TaskProvider, ReportSeriesProvider
from mlcomp.db.report_info import ReportLayoutInfo
from mlcomp.utils.io import yaml_load
from mlcomp.utils.misc import now
from mlcomp.db.models import ReportSeries
from mlcomp.utils.config import Config, merge_dicts_smart
from mlcomp.worker.executors.base import Executor
from mlcomp.contrib.catalyst.register import register


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
        return [
            (k, v) for k, v in self.__dict__.items() if not k.startswith('_')
        ]


# noinspection PyTypeChecker
@Executor.register
class Catalyst(Executor, Callback):
    __syn__ = 'catalyst'

    def __init__(
        self,
        args: Args,
        report: ReportLayoutInfo,
        cuda_devices: list,
        distr_info: dict,
        grid_config: dict,
    ):

        self.distr_info = distr_info
        self.args = args
        self.report = report
        self.experiment = None
        self.runner = None
        self.series_provider = ReportSeriesProvider()
        self.task_provider = TaskProvider()
        self.grid_config = grid_config
        self.master = True
        self.cuda_devices = cuda_devices
        self.checkpoint_resume = False

    def callbacks(self):
        result = dict()
        if self.master:
            result['catalyst'] = self

        return result

    def on_epoch_end(self, state: RunnerState):
        for s in self.report.series:
            train = state.metrics.epoch_values['train'][s.key]
            val = state.metrics.epoch_values['valid'][s.key]

            task_id = self.task.parent or self.task.id
            train = ReportSeries(
                part='train',
                name=s.key,
                epoch=state.epoch,
                task=task_id,
                value=train,
                time=now()
            )

            val = ReportSeries(
                part='valid',
                name=s.key,
                epoch=state.epoch,
                task=task_id,
                value=val,
                time=now()
            )

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
                    task = self.task
                    if task.parent:
                        task = self.task_provider.by_id(task.parent)
                    task.score = val.value
                    self.task_provider.update()

    def on_stage_start(self, state: RunnerState):
        state.loggers = [VerboseLogger(), RaiseExceptionLogger()]
        if self.checkpoint_resume:
            state.epoch += 1

    @classmethod
    def _from_config(
        cls, executor: dict, config: Config, additional_info: dict
    ):
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

        distr_info = additional_info.get('distr_info', {})
        cuda_devices = additional_info.get('cuda_devices')

        return cls(
            args=args,
            report=report,
            grid_config=grid_config,
            distr_info=distr_info,
            cuda_devices=cuda_devices
        )

    def set_dist_env(self, config):
        info = self.distr_info
        os.environ['MASTER_ADDR'] = info['master_addr']
        os.environ['MASTER_PORT'] = str(info['master_port'])
        os.environ['WORLD_SIZE'] = str(info['world_size'])

        os.environ['RANK'] = str(info['rank'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(info['local_rank'])

        config['distributed_params'] = {'rank': 0}

        if info['rank'] > 0:
            self.master = False

    def parse_args_uargs(self):
        args, config = parse_args_uargs(self.args, [])
        config = merge_dicts_smart(config, self.grid_config)

        if self.cuda_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = \
                ','.join(map(str, self.cuda_devices))

        if self.distr_info:
            self.set_dist_env(config)
        return args, config

    def _checkpoint_fix(self, callbacks: dict, logdir: str):
        path = os.path.join(logdir, 'checkpoints', 'best.pth')

        def mock(state):
            pass

        for c in callbacks.values():
            if isinstance(c, CheckpointCallback):
                if c.resume is None and os.path.exists(path):
                    c.resume = path

                if c.resume:
                    self.checkpoint_resume = True

                if not self.master:
                    c.on_epoch_end = mock
                    c.on_stage_end = mock

    def work(self):
        args, config = self.parse_args_uargs()
        set_global_seed(args.seed)

        Experiment, R = import_experiment_and_runner(Path(args.expdir))

        experiment = Experiment(config)
        runner: Runner = R()

        register()

        self.experiment = experiment
        self.runner = runner

        stages = experiment.stages[:]
        if self.distr_info:
            result = self.task.result
            if result:
                result = yaml_load(result)
                index = stages.index(result['stage']) + 1 \
                    if 'stage' in result else 0
            else:
                index = 0

            self.task.current_step = index + 1
            self.task.steps = len(stages)
            self.task_provider.commit()

            experiment.stages_config = {
                k: v
                for k, v in experiment.stages_config.items()
                if k == stages[index]
            }

        _get_callbacks = experiment.get_callbacks

        def get_callbacks(stage):
            res = _get_callbacks(stage)
            for k, v in self.callbacks().items():
                res[k] = v
            self._checkpoint_fix(res, experiment.logdir)

            return res

        experiment.get_callbacks = get_callbacks

        if experiment.logdir is not None:
            dump_config(config, experiment.logdir, args.configs)

        runner.run_experiment(experiment, check=args.check)

        return {'stage': experiment.stages[-1], 'stages': stages}


__all__ = ['Catalyst']
