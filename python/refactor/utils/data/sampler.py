# -*- coding: utf-8 -*-
#
# dataloader.py
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
from torch.utils.data import Sampler
import numpy as np
import torch as th


class SequentialEpochSampler(Sampler):
    def __init__(self, datasource, batch_size, max_step):
        self.datasource = datasource
        self._num_samples = batch_size * max_step
        super(SequentialEpochSampler, self).__init__(datasource)

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        repeat_num = (self._num_samples + len(self.datasource) - 1) // len(self.datasource)
        return iter(np.expand_dims(np.arange(len(self.datasource)), axis=0).repeat(repeat_num, axis=0).flatten()[:self._num_samples])

    def __len__(self):
        return self.num_samples

class LCWASampler:
    def __init__(self, num_nodes, device=th.device('cpu')):
        self.nodes = th.arange(num_nodes, device=device)

    def __iter__(self):
        return self

    def __next__(self):
        return {'negs': self.nodes}

class SequentialRandomSampler(Sampler):

    def __init__(self, data_source, batch_size, max_step,):
        self.data_source = data_source
        self.batch_size = batch_size
        self.max_step = max_step
        self._num_samples = self.batch_size * self.max_step

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        index_list = []
        for i in range(self.max_step):
            index_list += [np.random.choice(n, self.batch_size, replace=False)]
        index_list = np.concatenate(index_list, axis=None)
        return iter(index_list)

    def __len__(self):
        return self.num_samples


class SequentialTotalSampler(Sampler):
    def __init__(self, datasource, batch_size, max_step):
        self.datasource = datasource
        self._num_samples = batch_size * max_step

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        return iter(np.arange(self._num_samples))

    def __len__(self):
        return self.num_samples
