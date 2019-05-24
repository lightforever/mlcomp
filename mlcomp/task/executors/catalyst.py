from typing import List

from catalyst.dl.callbacks.core import Callback
from catalyst.dl.experiments import Experiment
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
from .report_info import *
import numpy as np
from scipy.special import softmax
import pickle
import cv2

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


# noinspection PyTypeChecker
@Executor.register
class Catalyst(Executor, Callback):
    def __init__(self, args: Args, report: ReportInfo):
        self.args = args
        self.series_provider = ReportSeriesProvider()
        self.img_provider = ReportImgProvider()
        self.report = report
        self.valid_data = {'input': [], 'output': [], 'target': []}
        self.experiment = None
        self.runner = None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name == 'train':
            return
        self.valid_data['input'].extend(state.input['features'].detach().cpu().numpy())
        self.valid_data['output'].extend(state.output['logits'].detach().cpu().numpy())
        self.valid_data['target'].extend(state.input['targets'].detach().cpu().numpy())

    def on_epoch_end(self, state: RunnerState):
        for s in self.report.series:
            train = state.metrics.epoch_values['train'][s.key]
            val = state.metrics.epoch_values['valid'][s.key]

            train = ReportSeries(group='train', name=s.name, epoch=state.epoch, task=self.task.id, value=train)
            val = ReportSeries(group='valid', name=s.name, epoch=state.epoch, task=self.task.id, value=val)

            self.series_provider.add(train)
            self.series_provider.add(val)

        for pr in self.report.precision_recall:
            output = np.array(self.valid_data['output'])
            output = softmax(output)[:, 1]
            img = pr.plot(self.valid_data['target'], output)
            content = {'img': img}
            obj = ReportImg(group=pr.name, epoch=state.epoch, task=self.task.id, img=pickle.dumps(content))
            self.img_provider.add(obj)

        for f1 in self.report.f1:
            output = np.array(self.valid_data['output'])
            img = f1.plot(self.valid_data['target'], output.argmax(1))
            content = {'img': img}
            obj = ReportImg(group=f1.name, epoch=state.epoch, task=self.task.id, img=pickle.dumps(content))
            self.img_provider.add(obj)

        for c in self.report.img_confusion:
            for i in range(c.count):
                content = self.valid_data['input'][i]
                content = self.experiment.denormilize(content)
                content = cv2.imencode('.jpg', content)[1].tostring()
                content = {'img': content}
                obj = ReportImg(group=c.name, epoch=state.epoch, task=self.task.id, img=pickle.dumps(content))
                self.img_provider.add(obj)

        self.valid_data = {'input': [], 'output': [], 'target': []}

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

        report_name = executor.get('report') or 'base'
        report = ReportInfo(config['reports'][report_name])
        return cls(args=args, report=report)

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

        self.experiment = experiment
        self.runner = runner

        runner.run_experiment(
            experiment,
            check=args.check
        )

