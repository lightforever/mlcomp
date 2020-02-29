from typing import Tuple

import numpy as np
from torch.nn import CrossEntropyLoss

from torch.nn.functional import cross_entropy
from torch.utils.data import Sampler

from catalyst.core import _State
from catalyst.dl import Callback


class HardNegativeSampler(Sampler, Callback):
    def __init__(
            self,
            data_source,
            name: str,
            count: int,
            batch_size: int = None,
            hard_interval: Tuple[float, float] = (50, 100),
            index_count: int = 1,
            criterion_data: dict = None
    ):
        super().__init__(data_source)
        self.criterion_data = criterion_data
        self.name = name
        self.order = 0
        self.hard_interval = hard_interval
        self.index_count = index_count

        self.sampled = 0
        self.count = count
        self.batch_size = batch_size or count
        self.max_index = len(data_source)
        self.loss = []
        self.indices = []

    def get_loss(self, state: _State, criterion=None, meta: dict = None):
        criterion = criterion or state.criterion
        if isinstance(criterion, dict):
            res = np.zeros(len(state.input['targets']))
            for k, v in criterion.items():
                res += self.get_loss(state, criterion=v,
                                     meta=self.criterion_data[k])
            return res

        output_key = 'logits' if meta is None else meta['output_key']
        input_key = 'targets' if meta is None else meta['input_key']

        if isinstance(criterion, CrossEntropyLoss):
            loss = cross_entropy(state.output[output_key],
                                 state.input[input_key],
                                 reduction='none'
                                 )
            loss = loss.cpu().detach().numpy()
        else:
            loss = []
            for i in range(len(state.input[input_key])):
                v = criterion(
                    state.output[output_key][i:i + 1],
                    state.input[input_key][i:i + 1]
                )
                loss.append(float(v))
            loss = np.array(loss)

        weight = 1 if meta is None else meta['weight']
        return loss * weight

    def __len__(self):
        return self.count

    def random(self, count: int = None):
        count = count or self.batch_size
        replace = count > self.max_index
        indices = []

        for i in range(self.index_count):
            index = np.random.choice(np.arange(0, self.max_index),
                                     replace=replace, size=count)
            indices.append(index)
        return indices

    def sample_batch(self):
        if len(self.loss) > 0:
            percentile_high = np.percentile(self.loss, self.hard_interval[1])
            percentile_low = np.percentile(self.loss, self.hard_interval[0])

            cond = (self.loss >= percentile_low) & (
                    self.loss <= percentile_high)
            hard_samples = np.where(cond)[0]
            indices = [index[hard_samples] for index in self.indices]
        else:
            indices = [[] for _ in range(self.index_count)]

        indices_random = self.random(self.batch_size - len(indices[0]))
        indices_shuffle = np.arange(0, self.batch_size)
        np.random.shuffle(indices_shuffle)

        for i in range(len(indices)):
            indices[i] = np.concatenate([indices[i], indices_random[i]])
            indices[i] = indices[i][indices_shuffle].astype(np.int32)

        if len(indices) == 1:
            return indices[0]
        return zip(indices)

    def __iter__(self):
        while self.sampled < self.count:
            batch = self.sample_batch()
            self.sampled += self.batch_size
            yield from batch

        self.sampled = 0

    def on_batch_end(self, state: _State):
        if state.loader_name != self.name:
            return

        self.loss = self.get_loss(state)
        self.indices = []

        for k in sorted(list(state.input)):
            if 'index_' in k:
                self.indices.append(state.input[k].cpu().detach().numpy())


class HardNegativePairSampler(HardNegativeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, index_count=2, **kwargs)


class HardNegativeTripleSampler(HardNegativeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, index_count=4, **kwargs)


class HardNegativeFourSampler(HardNegativeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, index_count=4, **kwargs)


__all__ = ['HardNegativeSampler', 'HardNegativePairSampler',
           'HardNegativeTripleSampler', 'HardNegativeFourSampler']
