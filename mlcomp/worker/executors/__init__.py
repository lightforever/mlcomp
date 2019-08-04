from .base import StepWrap, Executor
import os

if not os.getenv('SERVER'):
    # flake8: noqa
    from .kaggle import Download, Submit
    from .catalyst import Catalyst
    from .model import ModelAdd
    from .split import Split
