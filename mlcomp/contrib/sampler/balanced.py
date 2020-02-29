from typing import List, Iterator
import numpy as np
from torch.utils.data import Sampler


class BalanceClassSampler(Sampler):
    """
    Abstraction over data sampler. Allows you to create stratified sample
    on unbalanced classes.
    """

    def __init__(
            self,
            labels: List[int],
            mode: str = 'downsampling',
            count_per_class: List[int] = None,
            max_count: int = None
    ):
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

        if mode == 'upsampling' or max_count is not None:
            samples_per_class = max_count \
                if max_count is not None \
                else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        if count_per_class is not None:
            self.count_per_class = count_per_class
        else:
            self.count_per_class = [samples_per_class] * len(self.lbl2idx)
        self.labels = labels
        self.length = sum(self.count_per_class)

        super().__init__(labels)

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            samples = self.count_per_class[key]
            replace_ = samples > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], samples, replace=replace_
            ).tolist()
        assert (len(indices) == self.length)
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


__all__ = ['BalanceClassSampler']
