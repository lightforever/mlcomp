import os
import socket
from collections import OrderedDict
from pathlib import Path
from os.path import join

import torch

from catalyst.utils import set_global_seed, load_checkpoint
from catalyst.dl import RunnerState, Callback, Runner, CheckpointCallback
from catalyst.dl.callbacks import VerboseLogger, RaiseExceptionLogger
from catalyst.dl.utils.scripts import import_experiment_and_runner
from catalyst.utils.config import parse_args_uargs, dump_config

from mlcomp import TASK_FOLDER
from mlcomp.contrib.search.grid import grid_cells
from mlcomp.db.providers import ReportSeriesProvider
from mlcomp.db.report_info import ReportLayoutInfo
from mlcomp.utils.io import yaml_load, yaml_dump
from mlcomp.utils.misc import now
from mlcomp.db.models import ReportSeries
from mlcomp.utils.config import Config, merge_dicts_smart
from mlcomp.worker.executors.base import Executor
from mlcomp.contrib.catalyst.register import register
from mlcomp.worker.executors.model import trace_model_from_checkpoint
from mlcomp.worker.sync import copy_remote


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
    def __init__(
            self,
            args: Args,
            report: ReportLayoutInfo,
            cuda_devices: list,
            distr_info: dict,
            resume: dict,
            grid_config: dict,
            trace: str
    ):

        self.resume = resume
        self.distr_info = distr_info
        self.args = args
        self.report = report
        self.experiment = None
        self.runner = None
        self.series_provider = ReportSeriesProvider(self.session)
        self.grid_config = grid_config
        self.master = True
        self.cuda_devices = cuda_devices
        self.checkpoint_resume = False
        self.checkpoint_stage_epoch = 0
        self.trace = trace

    def callbacks(self):
        result = OrderedDict()
        if self.master:
            result['catalyst'] = self

        return result

    def on_epoch_start(self, state: RunnerState):
        if self.checkpoint_resume and state.stage_epoch == 0:
            state.epoch += 1

        state.stage_epoch = state.stage_epoch + self.checkpoint_stage_epoch
        state.checkpoint_data = {'stage_epoch': state.stage_epoch}
        if self.master:
            if state.stage_epoch == 0:
                self.step.start(1, name=state.stage)

            self.step.start(
                2, name=f'epoch {state.stage_epoch}', index=state.stage_epoch
            )

    def on_epoch_end(self, state: RunnerState):
        if self.master:
            self.step.end(2)

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
                time=now(),
                stage=state.stage
            )

            val = ReportSeries(
                part='valid',
                name=s.key,
                epoch=state.epoch,
                task=task_id,
                value=val,
                time=now(),
                stage=state.stage
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

    def on_stage_end(self, state: RunnerState):
        self.checkpoint_resume = False
        self.checkpoint_stage_epoch = 0
        if self.master:
            self.step.end(1)

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

        assert 'report_config' in additional_info, 'layout was not filled'
        report_config = additional_info['report_config']
        grid_cell = additional_info.get('grid_cell')
        report = ReportLayoutInfo(report_config)
        if len(args.configs) == 0:
            args.configs = [args.config]

        grid_config = {}
        if grid_cell is not None:
            grid_config = grid_cells(executor['grid'])[grid_cell][0]

        distr_info = additional_info.get('distr_info', {})
        cuda_devices = additional_info.get('cuda_devices')
        resume = additional_info.get('resume')

        return cls(
            args=args,
            report=report,
            grid_config=grid_config,
            distr_info=distr_info,
            cuda_devices=cuda_devices,
            resume=resume,
            trace=executor.get('trace')
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

    def _checkpoint_fix_config(self, experiment):
        resume = self.resume
        if not resume:
            return

        checkpoint_dir = join(experiment.logdir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        file = 'last.pth' if resume.get('load_last') else 'best.pth'

        path = join(checkpoint_dir, file)
        computer = socket.gethostname()
        if computer != resume['master_computer']:
            path_from = join(
                TASK_FOLDER, str(resume['master_task_id']), 'log',
                'checkpoints', file
            )
            self.info(
                f'copying checkpoint from: computer = '
                f'{resume["master_computer"]} path_from={path_from} '
                f'path_to={path}'
            )

            success = copy_remote(
                session=self.session,
                computer_from=resume['master_computer'],
                path_from=path_from,
                path_to=path
            )

            if not success:
                self.error(
                    f'copying from '
                    f'{resume["master_computer"]}/'
                    f'{path_from} failed'
                )
            else:
                self.info('checkpoint copied successfully')

        elif self.task.id != resume['master_task_id']:
            path = join(
                TASK_FOLDER, str(resume['master_task_id']), 'log',
                'checkpoints', file
            )
            self.info(
                f'master_task_id!=task.id, using checkpoint'
                f' from task_id = {resume["master_task_id"]}'
            )

        if not os.path.exists(path):
            self.info(f'no checkpoint at {path}')
            return

        ckpt = load_checkpoint(path)
        stages_config = experiment.stages_config
        for k, v in list(stages_config.items()):
            if k == ckpt['stage']:
                stage_epoch = ckpt['checkpoint_data']['stage_epoch'] + 1

                # if it is the last epoch in the stage
                if stage_epoch == v['state_params']['num_epochs'] \
                        or resume.get('load_best'):
                    del stages_config[k]
                    break

                self.checkpoint_stage_epoch = stage_epoch
                v['state_params']['num_epochs'] -= stage_epoch
                break
            del stages_config[k]

        stage = experiment.stages_config[experiment.stages[0]]
        for k, v in stage['callbacks_params'].items():
            if v.get('callback') == 'CheckpointCallback':
                v['resume'] = path

        self.info(f'found checkpoint at {path}')

    def _checkpoint_fix_callback(self, callbacks: dict):
        def mock(state):
            pass

        for k, c in callbacks.items():
            if not isinstance(c, CheckpointCallback):
                continue

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

        if self.master:
            task = self.task if not self.task.parent \
                else self.task_provider.by_id(self.task.parent)
            task.steps = len(stages)
            self.task_provider.commit()

        self._checkpoint_fix_config(experiment)

        _get_callbacks = experiment.get_callbacks

        def get_callbacks(stage):
            res = self.callbacks()
            for k, v in _get_callbacks(stage).items():
                res[k] = v

            self._checkpoint_fix_callback(res)
            return res

        experiment.get_callbacks = get_callbacks

        if experiment.logdir is not None:
            dump_config(config, experiment.logdir, args.configs)

        if self.distr_info:
            info = yaml_load(self.task.additional_info)
            info['resume'] = {
                'master_computer': self.distr_info['master_computer'],
                'master_task_id': self.task.id - self.distr_info['rank'],
                'load_best': True
            }
            self.task.additional_info = yaml_dump(info)
            self.task_provider.commit()

            experiment.stages_config = {
                k: v
                for k, v in experiment.stages_config.items()
                if k == experiment.stages[0]
            }

        runner.run_experiment(experiment, check=args.check)

        if self.master and self.trace:
            traced = trace_model_from_checkpoint(self.experiment.logdir, self)
            torch.jit.save(traced, self.trace)

        return {'stage': experiment.stages[-1], 'stages': stages}


__all__ = ['Catalyst']
