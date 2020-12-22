# -*- coding: utf-8 -*-
#
# tensor_models.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
KG Sparse embedding
"""
import os
import numpy as np
import copy

import torch as th
import torch.nn.init as INIT

from .ke_optimizer import *
from dglke.util import thread_wrapped_func
import torch.multiprocessing as mp
from torch.multiprocessing import Queue


class KGEmbedding:
    """Sparse Embedding for Knowledge Graph
    It is used to store both entity embeddings and relation embeddings.

    Parameters
    ----------
    num : int
        Number of embeddings.
    dim : int
        Embedding dimention size.
    device : th.device
        Device to store the embedding.
    """
    def __init__(self, device):
        self.emb = None
        self.is_train = False
        self.async_q = None
        self.device = device

    def init(self, emb_init, lr, async_threads, num=-1, dim=-1, init_strat='uniform', optimizer='Adagrad', device=None):
        """Initializing the embeddings for training.

        Parameters
        ----------
        emb_init : float or tuple
            The intial embedding range should be [-emb_init, emb_init].
        """
        self.async_threads = async_threads
        if device is not None:
            self.device = device
        if self.emb is None:
            self.emb = th.empty(num, dim, dtype=th.float32, device=self.device)
            self.num = self.emb.shape[0]
            self.dim = self.emb.shape[1]
        if optimizer == 'Adagrad':
            self.optim = Adagrad(self.emb, device=self.device, lr=lr)
        elif optimizer == 'Adam':
            self.optim = Adam(self.emb, device=self.device, lr=lr)
        else:
            raise NotImplementedError(f'optimizer {optimizer} is not supported by dglke yet.')

        self.trace = []
        self.has_cross_rel = False

        if init_strat == 'uniform':
            INIT.uniform_(self.emb, -emb_init, emb_init)
        elif init_strat == 'normal':
            if type(emb_init) is tuple or type(emb_init) is list:
                if len(emb_init) == 0:
                    mean = emb_init
                    std = 1
                else:
                    mean, std = emb_init
                INIT.normal_(self.emb.data, mean, std)
            else:
                init_size = emb_init
                INIT.normal_(self.emb.data)
                self.emb.data *= init_size
        elif init_strat == 'random':
            if type(emb_init) is tuple:
                x, y = emb_init
                self.emb.data = th.rand(num, dim, dtype=th.float32, device=self.device) * x + y
        elif init_strat == 'xavier':
            INIT.xavier_normal_(self.emb.data)
        elif init_strat == 'constant':
            INIT.constant_(self.emb.data, emb_init)

    def clone(self, device):
        clone_emb = copy.deepcopy(self)
        clone_emb.device = device
        clone_emb.emb = clone_emb.emb.to(device)
        clone_emb.optim = clone_emb.optim.to(device)
        return clone_emb

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name)
        self.emb = th.Tensor(np.load(file_name))

    def load_emb(self, emb_array):
        """Load embeddings from numpy array.

        Parameters
        ----------
        emb_array : numpy.array  or torch.tensor
            Embedding array in numpy array or torch.tensor
        """
        if isinstance(emb_array, np.ndarray):
            self.emb = th.Tensor(emb_array)
        else:
            self.emb = emb_array

    def save(self, path, name):
        """Save embeddings.

        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name)
        np.save(file_name, self.emb.cpu().detach().numpy())

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def setup_cross_rels(self, cross_rels, global_emb):
        cpu_bitmap = th.zeros((self.num,), dtype=th.bool)
        for i, rel in enumerate(cross_rels):
            cpu_bitmap[rel] = 1
        self.cpu_bitmap = cpu_bitmap
        self.has_cross_rel = True
        self.global_emb = global_emb

    def get_noncross_idx(self, idx):
        cpu_mask = self.cpu_bitmap[idx]
        gpu_mask = ~cpu_mask
        return idx[gpu_mask]

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process tensor access
        """
        self.emb.share_memory_()
        self.optim.share_memory()

    def __call__(self, idx, gpu_id=-1, trace=True):
        """ Return sliced tensor.

        Parameters
        ----------
        idx : th.tensor
            Slicing index
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
        # for inference or evaluation
        if self.is_train is False:
            return self.emb[idx].cuda(gpu_id, non_blocking=True)

        if self.has_cross_rel:
            cpu_idx = idx.cpu()
            cpu_mask = self.cpu_bitmap[cpu_idx]
            cpu_idx = cpu_idx[cpu_mask]
            cpu_idx = th.unique(cpu_idx)
            if cpu_idx.shape[0] != 0:
                cpu_emb = self.global_emb.emb[cpu_idx]
                self.emb[cpu_idx] = cpu_emb.cuda(gpu_id, non_blocking=True)
        s = self.emb[idx]
        if gpu_id >= 0:
            s = s.cuda(gpu_id, non_blocking=True)
        # During the training, we need to trace the computation.
        # In this case, we need to record the computation path and compute the gradients.
        if trace:
            data = s.clone().detach().requires_grad_(True)
            self.trace.append((idx, data))
        else:
            data = s
        return data

    def update(self, gpu_id=-1):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        with th.no_grad():
            for idx, data in self.trace:
                grad = data.grad.data
                # the update is non-linear so indices must be unique
                grad_indices = idx
                grad_values = grad
                if self.async_q is not None:
                    grad_indices.share_memory_()
                    grad_values.share_memory_()
                    self.async_q.put((grad_indices, grad_values, gpu_id))
                else:
                    if self.has_cross_rel:
                        cpu_mask = self.cpu_bitmap[grad_indices]
                        cpu_idx = grad_indices[cpu_mask]
                        if cpu_idx.shape[0] > 0:
                            cpu_grad = grad_values[cpu_mask]
                            self.global_emb.optim.step(cpu_idx, self.global_emb.emb, cpu_grad, gpu_id)
                    self.optim.step(grad_indices, self.emb, grad_values, gpu_id)
        self.trace = []

    def create_async_update(self):
        """Set up the async update subprocess.
        """
        self.async_q = Queue(1)
        self.async_p = mp.Process(target=self.async_update)
        self.async_p.start()

    def finish_async_update(self):
        """Notify the async update subprocess to quit.
        """
        self.async_q.put((None, None, None))
        self.async_p.join()

    def async_update(self):
        th.set_num_threads(self.async_threads)
        while True:
            (grad_indices, grad_values, gpu_id) = self.async_q.get()
            if grad_indices is None:
                return
            with th.no_grad():
                if self.has_cross_rel:
                    cpu_mask = self.cpu_bitmap[grad_indices]
                    cpu_idx = grad_indices[cpu_mask]
                    if cpu_idx.shape[0] > 0:
                        cpu_grad = grad_values[cpu_mask]
                        self.global_emb.optim.step(cpu_idx, self.global_emb.emb, cpu_grad, gpu_id)
                self.optim.step(grad_indices, self.emb, grad_values, gpu_id)


    def curr_emb(self):
        """Return embeddings in trace.
        """
        data = [data for _, data in self.trace]
        return th.cat(data, 0)

