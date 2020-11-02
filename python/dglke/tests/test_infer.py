# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from __future__ import absolute_import

import os
import scipy as sp
import numpy as np
import dgl.backend as F
import dgl

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import mxnet as mx
    mx.random.seed(42)
    np.random.seed(42)

    from models.mxnet.score_fun import *
    from models.mxnet.tensor_models import ExternalEmbedding
else:
    import torch as th
    th.manual_seed(42)
    np.random.seed(42)

    from models.pytorch.score_fun import *
    from models.pytorch.tensor_models import ExternalEmbedding

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def generate_rand_emb(func_name, bcast):
    dim=16

    num_head = 16
    num_rels = 4
    num_tail = 32
    if bcast == 'rel':
        num_rels = 1
    if bcast == 'head':
        num_head = 1
    if bcast == 'tail':
        num_tail = 1

    head_emb = F.uniform((num_head, dim), F.float32, F.cpu(), 0, 1)
    tail_emb = F.uniform((num_tail, dim), F.float32, F.cpu(), 0, 1)
    rel_emb = F.uniform((num_rels, dim), F.float32, F.cpu(), -1, 1)

    if func_name == 'RotatE':
        rel_emb = F.uniform((num_rels, dim//2), F.float32, F.cpu(), -1, 1)
    if func_name == 'RESCAL':
        rel_emb = F.uniform((num_rels, dim * dim), F.float32, F.cpu(), -1, 1)

    if func_name == 'TransE':
        return head_emb, rel_emb, tail_emb, (12.0)
    elif func_name == 'TransE_l1':
        return head_emb, rel_emb, tail_emb, (12.0, 'l1')
    elif func_name == 'TransE_l2':
        return head_emb, rel_emb, tail_emb, (12.0, 'l2')
    elif func_name == 'RESCAL':
        return head_emb, rel_emb, tail_emb, (dim, dim)
    elif func_name == 'RotatE':
        return head_emb, rel_emb, tail_emb, (12.0, 1.0)
    elif func_name == 'SimplE':
        return head_emb, rel_emb, tail_emb, None
    else:
        return head_emb, rel_emb, tail_emb, None

ke_infer_funcs = {'TransE': TransEScore,
                  'TransE_l1': TransEScore,
                  'TransE_l2': TransEScore,
                  'DistMult': DistMultScore,
                  'ComplEx': ComplExScore,
                  'RESCAL': RESCALScore,
                  'TransR': TransRScore,
                  'RotatE': RotatEScore,
                  'SimplE': SimplEScore}

class FakeEdge:
    def __init__(self, hemb, temb, remb):
        data = {}
        data['head_emb'] = hemb
        data['tail_emb'] = temb
        data['emb'] = remb
        src = {}
        src['emb'] = hemb
        dst = {}
        dst['emb'] = temb

        self._src = src
        self._dst = dst
        self._data = data

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def data(self):
        return self._data

class InferKEModel:
    def __init__(self, score_func, entity_emb, rel_emb):
        self.score_func = score_func
        self.entity_emb = entity_emb
        self.rel_emb = rel_emb

    def infer_score(self):
        head_emb = generate_rand_emb

def check_infer_score(func_name):
    batch_size = 10

    ke_score_func = ke_infer_funcs[func_name]

    # normal
    head_emb, rel_emb, tail_emb, args = generate_rand_emb(func_name, 'none')
    if args is None:
        score_func = ke_score_func()
    elif type(args) is tuple:
        score_func = ke_score_func(*list(args))
    else:
        score_func = ke_score_func(args)
    score1 = score_func.infer(head_emb, rel_emb, tail_emb)
    assert(score1.shape[0] == head_emb.shape[0])
    h_score = []
    for i in range(head_emb.shape[0]):
        r_score = []
        for j in range(rel_emb.shape[0]):
            t_score = []
            for k in range(tail_emb.shape[0]):
                hemb = head_emb[i]
                remb = rel_emb[j]
                temb = F.unsqueeze(tail_emb[k], dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = score_func.edge_func(edge)['score']
                t_score.append(F.asnumpy(score))
            r_score.append(t_score)
        h_score.append(r_score)
    score2 = np.asarray(h_score).reshape(head_emb.shape[0], rel_emb.shape[0], tail_emb.shape[0])
    np.testing.assert_allclose(F.asnumpy(score1), score2,
                                   rtol=1e-5, atol=1e-5)

    # bcast head
    head_emb, rel_emb, tail_emb, args = generate_rand_emb(func_name, 'head')
    if args is None:
        score_func = ke_score_func()
    elif type(args) is tuple:
        score_func = ke_score_func(*list(args))
    else:
        score_func = ke_score_func(args)
    score1 = score_func.infer(head_emb, rel_emb, tail_emb)
    assert(score1.shape[0] == head_emb.shape[0])
    h_score = []
    for i in range(head_emb.shape[0]):
        r_score = []
        for j in range(rel_emb.shape[0]):
            t_score = []
            for k in range(tail_emb.shape[0]):
                hemb = head_emb[i]
                remb = rel_emb[j]
                temb = F.unsqueeze(tail_emb[k], dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = score_func.edge_func(edge)['score']
                t_score.append(F.asnumpy(score))
            r_score.append(t_score)
        h_score.append(r_score)
    score2 = np.asarray(h_score).reshape(1, rel_emb.shape[0], tail_emb.shape[0])
    np.testing.assert_allclose(F.asnumpy(score1), score2,
                                   rtol=1e-5, atol=1e-5)

    # bcast rel
    head_emb, rel_emb, tail_emb, args = generate_rand_emb(func_name, 'rel')
    if args is None:
        score_func = ke_score_func()
    elif type(args) is tuple:
        score_func = ke_score_func(*list(args))
    else:
        score_func = ke_score_func(args)
    score1 = score_func.infer(head_emb, rel_emb, tail_emb)
    assert(score1.shape[0] == head_emb.shape[0])
    h_score = []
    for i in range(head_emb.shape[0]):
        r_score = []
        for j in range(rel_emb.shape[0]):
            t_score = []
            for k in range(tail_emb.shape[0]):
                hemb = head_emb[i]
                remb = rel_emb[j]
                temb = F.unsqueeze(tail_emb[k], dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = score_func.edge_func(edge)['score']
                t_score.append(F.asnumpy(score))
            r_score.append(t_score)
        h_score.append(r_score)
    score2 = np.asarray(h_score).reshape(head_emb.shape[0], 1, tail_emb.shape[0])
    np.testing.assert_allclose(F.asnumpy(score1), score2,
                                   rtol=1e-5, atol=1e-5)

    # bcast tail
    head_emb, rel_emb, tail_emb, args = generate_rand_emb(func_name, 'tail')
    if args is None:
        score_func = ke_score_func()
    elif type(args) is tuple:
        score_func = ke_score_func(*list(args))
    else:
        score_func = ke_score_func(args)
    score1 = score_func.infer(head_emb, rel_emb, tail_emb)
    assert(score1.shape[0] == head_emb.shape[0])
    h_score = []
    for i in range(head_emb.shape[0]):
        r_score = []
        for j in range(rel_emb.shape[0]):
            t_score = []
            for k in range(tail_emb.shape[0]):
                hemb = head_emb[i]
                remb = rel_emb[j]
                temb = F.unsqueeze(tail_emb[k], dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = score_func.edge_func(edge)['score']
                t_score.append(F.asnumpy(score))
            r_score.append(t_score)
        h_score.append(r_score)
    score2 = np.asarray(h_score).reshape(head_emb.shape[0], rel_emb.shape[0], 1)
    np.testing.assert_allclose(F.asnumpy(score1), score2,
                                   rtol=1e-5, atol=1e-5)

def test_score_func_transe():
    check_infer_score('TransE')
    check_infer_score('TransE_l1')
    check_infer_score('TransE_l2')

def test_score_func_distmult():
    check_infer_score('DistMult')

def test_score_func_complex():
    check_infer_score('ComplEx')

def test_score_func_rescal():
    check_infer_score('RESCAL')

def test_score_func_rotate():
    check_infer_score('RotatE')

def test_score_func_simple():
    check_infer_score('SimplE')

if __name__ == '__main__':
    test_score_func_transe()
    test_score_func_distmult()
    test_score_func_complex()
    test_score_func_rescal()
    test_score_func_rotate()
    test_score_func_simple()