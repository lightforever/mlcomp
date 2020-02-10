from typing import List
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


class BalancedClassSamplerDistributed(DistributedSamplerIndices):
    """
    Abstraction over data sampler. Allows you to create stratified sample
    on unbalanced classes.
    """

    def __init__(self, labels: List[int], mode: str = "downsampling"):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the datasety
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum()
            for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = mode \
                if isinstance(mode, int) \
                else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

        super().__init__(DatasetDummy(self.length))

    def get_indices(self):
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert (len(indices) == self.length)
        np.random.shuffle(indices)

        return list(iter(indices))


__all__ = ['BalancedClassSamplerDistributed', 'DistributedSamplerIndices']
