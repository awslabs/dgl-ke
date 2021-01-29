from .optimizer import Optimizer
import itertools
import torch as th

class Adagrad(Optimizer):
    def __init__(self, dense_params, sparse_params, lr=1e-3, eps=1e-8):
        super(Adagrad, self).__init__()
        self.dense_optim = th.optim.Adagrad(itertools.chain(dense_params, sparse_params), lr=lr, eps=eps)
