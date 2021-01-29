import torch as th
from .optimizer import Optimizer
class Adam(Optimizer):
    def __init__(self, dense_params, sparse_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(Adam, self).__init__()
        self.dense_optim = th.optim.Adam(dense_params, lr, betas, eps, weight_decay)
        self.sparse_optim = SparseAdamOptimizer(sparse_params, lr, betas, eps, weight_decay)
