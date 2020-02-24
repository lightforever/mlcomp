import numpy as np

from torch.utils.data import DistributedSampler, Dataset


class DatasetDummy(Dataset):
    def __init__(self, count: int):
        self.count = count

    def __len__(self):
        return self.count


class DistributedSamplerIndices(DistributedSampler):
    def get_indices(self):
        return list(self.__iter__())

    def __iter__(self):
        # deterministically shuffle based on epoch
        np.random.seed(self.epoch)

        indices = self.get_indices()

        if self.shuffle:
            np.random.shuffle(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


__all__ = ['DistributedSamplerIndices', 'DatasetDummy']
