import numpy as np

from mlcomp.worker.executors import Executor

from dataset import MnistDataset
from mlcomp.worker.interfaces import Torch


@Executor.register
class Infer(Executor):
    def work(self):
        dataset = MnistDataset(file='data/test.csv')
        torch_interface = Torch('models/net.pth', 128, name='net')
        prob = torch_interface({'dataset': dataset})['prob']
        np.save(f'data/net_test', prob)
