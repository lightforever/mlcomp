from torch.utils.data import Sampler


class HardNegativeSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)

    def __len__(self):
        pass

    def __iter__(self):
        pass
