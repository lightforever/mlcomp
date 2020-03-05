import os
from os.path import join
import shutil
from pathlib import Path
import sys

import safitty
import torch
from catalyst.dl import Runner
from torch.jit import ScriptModule
import torch.nn as nn

from catalyst.dl.core import Experiment
from catalyst import utils
from catalyst.utils import import_experiment_and_runner

from mlcomp import TASK_FOLDER, MODEL_FOLDER
from mlcomp.db.models import Model
from mlcomp.db.providers import TaskProvider, ModelProvider, \
    ProjectProvider
from mlcomp.utils.misc import now
from mlcomp.utils.config import Config
from mlcomp.worker.executors import Executor


class _ForwardOverrideModel(nn.Module):
    """
    Model that calls specified method instead of forward

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method = method_name

    def forward(self, *args, **kwargs):
        args = args[0][self.method]
        if isinstance(args, dict):
            kwargs = args
            args = ()
        return getattr(self.model, self.method)(*args, **kwargs)


class _TracingModelWrapper(nn.Module):
    """
    Wrapper that traces model with batch instead of calling it

    (Workaround, to use native model batch handler)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.method_name = method_name
        self.model = model
        self.tracing_result: ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = _ForwardOverrideModel(self.model, self.method_name)
        example_inputs = {
            self.method_name: kwargs if len(kwargs) > 0 else args
        }

        # noinspection PyTypeChecker
        self.tracing_result = torch.jit.trace(
            method_model, example_inputs=example_inputs
        )


def trace_model(
    model: Model,
    runner: Runner,
    batch=None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: str = "cpu",
    predict_params: dict = None,
) -> ScriptModule:
    """
    Traces model using runner and batch

    Args:
        model: Model to trace
        runner: Model's native runner that was used to train model
        batch: Batch to trace the model
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): Apex FP16 init level, optional
        device (str): Torch device
        predict_params (dict): additional parameters for model forward

    Returns:
        (ScriptModule): Traced model
    """
    if batch is None or runner is None:
        raise ValueError("Both batch and runner must be specified.")

    if mode not in ["train", "eval"]:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'eval' or 'train'")

    predict_params = predict_params or {}

    tracer = _TracingModelWrapper(model, method_name)
    if opt_level is not None:
        utils.assert_fp16_available()
        # If traced in AMP we need to initialize the model before calling
        # the jit
        # https://github.com/NVIDIA/apex/issues/303#issuecomment-493142950
        from apex import amp
        model = model.to(device)
        model = amp.initialize(model, optimizers=None, opt_level=opt_level)
        # after fixing this bug https://github.com/pytorch/pytorch/issues/23993
        params = {**predict_params, "check_trace": False}
    else:
        params = predict_params

    getattr(model, mode)()
    utils.set_requires_grad(model, requires_grad=requires_grad)

    _runner_model, _runner_device = runner.model, runner.device

    runner.model, runner.device = tracer, device
    runner.predict_batch(batch, **params)
    result: ScriptModule = tracer.tracing_result

    runner.model, runner.device = _runner_model, _runner_device
    return result


def trace_model_from_checkpoint(logdir, logger, method_name='forward',
                                file='best'):
    config_path = f'{logdir}/configs/_config.json'
    checkpoint_path = f'{logdir}/checkpoints/{file}.pth'
    logger.info('Load config')
    config = safitty.load(config_path)
    if 'distributed_params' in config:
        del config['distributed_params']

    # Get expdir name
    # noinspection SpellCheckingInspection,PyTypeChecker
    # We will use copy of expdir from logs for reproducibility
    expdir_name = config['args']['expdir']
    logger.info(f'expdir_name from args: {expdir_name}')

    sys.path.insert(0, os.path.abspath(join(logdir, '../')))

    expdir_from_logs = os.path.abspath(join(logdir, '../', expdir_name))

    logger.info(f'expdir_from_logs: {expdir_from_logs}')
    logger.info('Import experiment and runner from logdir')

    ExperimentType, RunnerType = \
        import_experiment_and_runner(Path(expdir_from_logs))
    experiment: Experiment = ExperimentType(config)

    logger.info(f'Load model state from checkpoints/{file}.pth')
    model = experiment.get_model(next(iter(experiment.stages)))
    checkpoint = utils.load_checkpoint(checkpoint_path)
    utils.unpack_checkpoint(checkpoint, model=model)

    device = 'cpu'
    stage = list(experiment.stages)[0]
    loader = 0
    mode = 'eval'
    requires_grad = False
    opt_level = None

    runner: RunnerType = RunnerType()
    runner.model, runner.device = model, device

    batch = experiment.get_native_batch(stage, loader)

    logger.info('Tracing')
    traced = trace_model(
        model,
        runner,
        batch,
        method_name=method_name,
        mode=mode,
        requires_grad=requires_grad,
        opt_level=opt_level,
        device=device,
    )

    logger.info('Done')
    return traced


@Executor.register
class ModelAdd(Executor):
    def __init__(
            self,
            name: str,
            project: int,
            fold: int,
            train_task: int = None,
            child_task: int = None,
            file: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.train_task = train_task
        self.name = name
        self.child_task = child_task
        self.project = project
        self.file = file
        self.fold = fold

    def work(self):
        project = ProjectProvider(self.session).by_id(self.project)

        self.info(f'Task = {self.train_task} child_task: {self.child_task}')

        model = Model(
            created=now(),
            name=self.name,
            project=self.project,
            equations='',
            fold=self.fold
        )

        provider = ModelProvider(self.session)
        if self.train_task:
            task_provider = TaskProvider(self.session)
            task = task_provider.by_id(self.train_task)
            model.score_local = task.score

            task_dir = join(TASK_FOLDER, str(self.child_task or task.id))
            src_log = f'{task_dir}/logs'
            models_dir = join(MODEL_FOLDER, project.name)
            os.makedirs(models_dir, exist_ok=True)

            model_path_tmp = f'{src_log}/traced.pth'
            traced = trace_model_from_checkpoint(src_log, self, file=self.file)

            model_path = f'{models_dir}/{model.name}.pth'
            model_weight_path = f'{models_dir}/{model.name}_weight.pth'
            torch.jit.save(traced, model_path_tmp)
            shutil.copy(model_path_tmp, model_path)
            file = self.file = 'best_full'
            shutil.copy(f'{src_log}/checkpoints/{file}.pth',
                        model_weight_path)

        provider.add(model)

    @classmethod
    def _from_config(
            cls, executor: dict, config: Config, additional_info: dict
    ):
        return ModelAdd(
            name=executor['name'],
            project=executor['project'],
            train_task=executor['task'],
            child_task=executor['child_task'],
            fold=executor['fold'],
            file=executor['file']
        )


__all__ = ['ModelAdd', 'trace_model_from_checkpoint']

if __name__ == '__main__':
    import logging
    trace_model_from_checkpoint(
        '/home/ingenix/mlcomp/tasks/64913/log',
        logging
    )
