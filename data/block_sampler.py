import math
import numpy as np
from torch.utils.data.sampler import Sampler


class BlockSampler(Sampler):
    def __init__(self, indices, n_a):
        n_b = math.ceil(len(indices) / n_a)
        pad = n_a * n_b - len(indices)
        self.idx_arr = np.pad(np.arange(len(indices)), [0, pad],
                              'maximum').reshape([n_a, n_b])
        np.random.shuffle(self.idx_arr)

    def __iter__(self):
        return iter(self.idx_arr.reshape(-1).tolist())


class BlockSubsetSampler(BlockSampler):
    def __init__(self, indices, n_a, n_block):
        super().__init__(indices, n_a)
        self.k = n_block

    def __iter__(self):
        return iter(self.idx_arr[:self.k].reshape(-1).tolist())
