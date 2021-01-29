import torch as th
from torch.nn import Embedding
import numpy as np
import copy
import os


class KGEmbedding:
    def __init__(self):
        self.emb = None
        self.is_train = False
        self.async_q = None
        self.device = None

    def init(self, num=-1, dim=-1, init_func=None, device=None, backend='torch'):
        if device is not None:
            self.device = device
        if self.emb is None:
            if backend == 'torch':
                self.emb = Embedding(num_embeddings=num,
                                     embedding_dim=dim,
                                     sparse=False).to(self.device)
            if init_func is not None:
                init_func(self.emb.weight)
            else:
                raise NotImplementedError(f'backend {backend} is not supported.')

    def clone(self, device):
        clone_emb = copy.deepcopy(self)
        clone_emb.device = device
        clone_emb.emb = clone_emb.emb.to(device)
        return clone_emb

    def load(self, path, name):
        file_name = os.path.join(path, name)
        self.emb.load_state_dict(th.load(file_name, map_location={'cpu': self.device}))

    def load_emb(self, emb_array):
        if isinstance(emb_array, np.ndarray):
            self.emb = th.Tensor(emb_array)
        else:
            self.emb = emb_array

    def save(self, path, name):
        file_name = os.path.join(path, name)
        th.save(self.emb.cpu().detach().state_dict(), file_name)

    def train(self):
        self.is_train = True
        self.emb.train()

    def eval(self):
        self.is_train = False
        self.emb.eval()

    def share_memory(self):
        self.emb.share_memory_()

    def to(self, gpu_id=-1):
        device = th.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        self.emb
    def __call__(self, idx):
        emb = self.emb(idx)
        return emb

    def parameters(self):
        return self.emb.parameters()
