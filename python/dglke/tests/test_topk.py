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

import os
import scipy as sp
import dgl
import numpy as np
import dgl.backend as F
import dgl

from models import InferModel
from models.infer import ScoreInfer, EmbSimInfer

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import mxnet as mx
    mx.random.seed(40)
    np.random.seed(40)
    from models.mxnet.tensor_models import InferEmbedding
    from models.mxnet.tensor_models import norm
else:
    import torch as th
    th.manual_seed(42)
    np.random.seed(42)
    from models.pytorch.tensor_models import InferEmbedding
    from models.pytorch.tensor_models import norm

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def generate_rand_emb(func_name, num_entity, num_rels, dim, bcast):
    if bcast == 'rel':
        num_rels = 1
    if bcast == 'head':
        num_head = 1
    if bcast == 'tail':
        num_tail = 1

    entity_emb = F.uniform((num_entity, dim), F.float32, F.cpu(), -1, 1)
    rel_emb = F.uniform((num_rels, dim), F.float32, F.cpu(), -1, 1)

    if func_name == 'RotatE':
        rel_emb = F.uniform((num_rels, dim//2), F.float32, F.cpu(), -1, 1)
    if func_name == 'RESCAL':
        rel_emb = F.uniform((num_rels, dim * dim), F.float32, F.cpu(), -1, 1)

    if func_name == 'TransE':
        return entity_emb, rel_emb
    elif func_name == 'TransE_l1':
        return entity_emb, rel_emb
    elif func_name == 'TransE_l2':
        return entity_emb, rel_emb
    elif func_name == 'RESCAL':
        return entity_emb, rel_emb
    elif func_name == 'RotatE':
        return entity_emb, rel_emb
    else:
        return entity_emb, rel_emb

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

def check_topk_score(model_name):
    hidden_dim = 32
    gamma = 12.0

    num_entity = 40
    num_rels = 4
    score_model = ScoreInfer(-1, 'config', 'path', 'none')
    if model_name == 'TransE' or \
        model_name =='TransE_l1' or \
        model_name == 'TransE_l2' or \
        model_name == 'DistMult' or \
        model_name == 'ComplEx':
        model = InferModel('cpu', model_name, hidden_dim, batch_size=16)
    elif model_name == 'RESCAL':
        model = InferModel('cpu', model_name, hidden_dim)
    elif model_name == 'RotatE':
        model = InferModel('cpu', model_name, hidden_dim, double_entity_emb=True)
        
    entity_emb, rel_emb = generate_rand_emb(model_name, num_entity, num_rels, hidden_dim, 'none')
    model.entity_emb = InferEmbedding('cpu')
    model.entity_emb.emb = entity_emb
    model.relation_emb = InferEmbedding('cpu')
    model.relation_emb.emb = rel_emb
    score_model.model = model
    score_func = model.score_func

    head = F.arange(0, num_entity // 2)
    rel = F.arange(0, num_rels)
    tail = F.arange(num_entity // 2, num_entity)

    # exec_model==triplet_wise
    tw_rel = np.random.randint(0, num_rels, num_entity // 2)
    tw_rel = F.tensor(tw_rel)
    result1 = score_model.topK(head, tw_rel, tail, exec_mode='triplet_wise')
    assert len(result1) == 1
    scores = []
    head_ids = []
    rel_ids = []
    tail_ids = []
    for i in range(head.shape[0]):
        hemb = F.take(entity_emb, head[i], 0)
        remb = F.take(rel_emb, tw_rel[i], 0)
        temb = F.unsqueeze(F.take(entity_emb, tail[i], 0), dim=0)
        edge = FakeEdge(hemb, temb, remb)
        score = F.asnumpy(score_func.edge_func(edge)['score'])
        scores.append(score)
        head_ids.append(F.asnumpy(head[i]))
        rel_ids.append(F.asnumpy(tw_rel[i]))
        tail_ids.append(F.asnumpy(tail[i]))
    scores = np.asarray(scores)
    scores = scores.reshape(scores.shape[0])
    head_ids = np.asarray(head_ids)
    rel_ids = np.asarray(rel_ids)
    tail_ids = np.asarray(tail_ids)
    idx = np.argsort(scores)
    idx = idx[::-1]
    idx = idx[:10]
    head_ids = head_ids[idx]
    rel_ids = rel_ids[idx]
    tail_ids = tail_ids[idx]
    score_topk = scores[idx]

    r1_head, r1_rel, r1_tail, r1_score = result1[0]
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r1_rel, rel_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)

    # exec_mode==all
    result1 = score_model.topK(head, rel, tail, k=20)
    result2 = score_model.topK(head=head, tail=tail, k=20)
    assert len(result1) == 1
    assert len(result2) == 1

    scores = []
    head_ids = []
    rel_ids = []
    tail_ids = []
    for i in range(head.shape[0]):
        for j in range(rel.shape[0]):
            for k in range(tail.shape[0]):
                hemb = F.take(entity_emb, head[i], 0)
                remb = F.take(rel_emb, rel[j], 0)
                temb = F.unsqueeze(F.take(entity_emb, tail[k], 0), dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = F.asnumpy(score_func.edge_func(edge)['score'])
                scores.append(score)
                head_ids.append(F.asnumpy(head[i]))
                rel_ids.append(F.asnumpy(rel[j]))
                tail_ids.append(F.asnumpy(tail[k]))

    scores = np.asarray(scores)
    scores = scores.reshape(scores.shape[0])
    head_ids = np.asarray(head_ids)
    rel_ids = np.asarray(rel_ids)
    tail_ids = np.asarray(tail_ids)
    idx = np.argsort(scores)
    idx = idx[::-1]
    idx = idx[:20]
    head_ids = head_ids[idx]
    rel_ids = rel_ids[idx]
    tail_ids = tail_ids[idx]
    score_topk = scores[idx]

    r1_head, r1_rel, r1_tail, r1_score = result1[0]
    r2_head, r2_rel, r2_tail, r2_score = result2[0]
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r2_head, head_ids)
    np.testing.assert_allclose(r1_rel, rel_ids)
    np.testing.assert_allclose(r2_rel, rel_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)
    np.testing.assert_allclose(r2_tail, tail_ids)

    result1 = score_model.topK(head, rel, tail, exec_mode='batch_rel')
    result2 = score_model.topK(head=head, tail=tail, exec_mode='batch_rel')
    assert len(result1) == num_rels
    assert len(result2) == num_rels
    for j in range(rel.shape[0]):
        scores = []
        head_ids = []
        rel_ids = []
        tail_ids = []
        for i in range(head.shape[0]):
            for k in range(tail.shape[0]):
                hemb = F.take(entity_emb, head[i], 0)
                remb = F.take(rel_emb, rel[j], 0)
                temb = F.unsqueeze(F.take(entity_emb, tail[k], 0), dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = F.asnumpy(score_func.edge_func(edge)['score'])
                scores.append(score)
                head_ids.append(F.asnumpy(head[i]))
                rel_ids.append(F.asnumpy(rel[j]))
                tail_ids.append(F.asnumpy(tail[k]))

        scores = np.asarray(scores)
        scores = scores.reshape(scores.shape[0])
        head_ids = np.asarray(head_ids)
        rel_ids = np.asarray(rel_ids)
        tail_ids = np.asarray(tail_ids)
        idx = np.argsort(scores)
        idx = idx[::-1]
        idx = idx[:10]
        head_ids = head_ids[idx]
        rel_ids = rel_ids[idx]
        tail_ids = tail_ids[idx]
        score_topk = scores[idx]

        r1_head, r1_rel, r1_tail, r1_score = result1[j]
        r2_head, r2_rel, r2_tail, r2_score = result2[j]
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r1_rel, rel_ids)
        np.testing.assert_allclose(r2_rel, rel_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)

    head = F.arange(0, num_entity)
    rel = F.arange(0, num_rels)
    tail = F.arange(0, num_entity)
    result1 = score_model.topK(head, rel, tail, exec_mode='batch_head')
    result2 = score_model.topK(exec_mode='batch_head')
    assert len(result1) == num_entity
    assert len(result2) == num_entity

    for i in range(head.shape[0]):
        scores = []
        head_ids = []
        rel_ids = []
        tail_ids = []
        for j in range(rel.shape[0]):
            for k in range(tail.shape[0]):
                hemb = F.take(entity_emb, head[i], 0)
                remb = F.take(rel_emb, rel[j], 0)
                temb = F.unsqueeze(F.take(entity_emb, tail[k], 0), dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = F.asnumpy(score_func.edge_func(edge)['score'])
                scores.append(score)
                head_ids.append(F.asnumpy(head[i]))
                rel_ids.append(F.asnumpy(rel[j]))
                tail_ids.append(F.asnumpy(tail[k]))

        scores = np.asarray(scores)
        scores = scores.reshape(scores.shape[0])
        head_ids = np.asarray(head_ids)
        rel_ids = np.asarray(rel_ids)
        tail_ids = np.asarray(tail_ids)
        idx = np.argsort(scores)
        idx = idx[::-1]
        idx = idx[:10]
        head_ids = head_ids[idx]
        rel_ids = rel_ids[idx]
        tail_ids = tail_ids[idx]
        score_topk = scores[idx]

        r1_head, r1_rel, r1_tail, r1_score = result1[i]
        r2_head, r2_rel, r2_tail, r2_score = result2[i]
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r1_rel, rel_ids)
        np.testing.assert_allclose(r2_rel, rel_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)

    result1 = score_model.topK(head, rel, tail, exec_mode='batch_tail')
    result2 = score_model.topK(exec_mode='batch_tail')
    assert len(result1) == num_entity
    assert len(result2) == num_entity
    for k in range(tail.shape[0]):
        scores = []
        head_ids = []
        rel_ids = []
        tail_ids = []
        for i in range(head.shape[0]):
            for j in range(rel.shape[0]):
                hemb = F.take(entity_emb, head[i], 0)
                remb = F.take(rel_emb, rel[j], 0)
                temb = F.unsqueeze(F.take(entity_emb, tail[k], 0), dim=0)
                edge = FakeEdge(hemb, temb, remb)
                score = F.asnumpy(score_func.edge_func(edge)['score'])
                scores.append(score)
                head_ids.append(F.asnumpy(head[i]))
                rel_ids.append(F.asnumpy(rel[j]))
                tail_ids.append(F.asnumpy(tail[k]))

        scores = np.asarray(scores)
        scores = scores.reshape(scores.shape[0])
        head_ids = np.asarray(head_ids)
        rel_ids = np.asarray(rel_ids)
        tail_ids = np.asarray(tail_ids)
        idx = np.argsort(scores)
        idx = idx[::-1]
        idx = idx[:10]
        head_ids = head_ids[idx]
        rel_ids = rel_ids[idx]
        tail_ids = tail_ids[idx]
        score_topk = scores[idx]

        r1_head, r1_rel, r1_tail, r1_score = result1[k]
        r2_head, r2_rel, r2_tail, r2_score = result2[k]
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r1_rel, rel_ids)
        np.testing.assert_allclose(r2_rel, rel_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-6, atol=1e-6)

def test_topk_transe():
    check_topk_score('TransE')
    check_topk_score('TransE_l1')
    check_topk_score('TransE_l2')

def test_topk_distmult():
    check_topk_score('DistMult')

def test_topk_complex():
    check_topk_score('ComplEx')

def test_topk_rescal():
    check_topk_score('RESCAL')

def test_topk_rotate():
    check_topk_score('RotatE')

def run_topk_emb(sfunc, sim_func):
    hidden_dim = 32
    num_head = 40
    num_tail = 40
    num_emb = 80

    emb = F.uniform((num_emb, hidden_dim), F.float32, F.cpu(), -1, 1)
    head = F.arange(0, num_head)
    tail = F.arange(num_head, num_head+num_tail)
    sim_infer = EmbSimInfer(-1, None, sfunc, 32)
    sim_infer.emb = emb

    result1 = sim_infer.topK(head, tail, pair_ws=True)
    scores = []
    head_ids = []
    tail_ids = []
    for i in range(head.shape[0]):
        j = i
        hemb = F.take(emb, head[i], 0)
        temb = F.take(emb, tail[j], 0)

        score = sim_func(hemb, temb)
        scores.append(F.asnumpy(score))
        head_ids.append(F.asnumpy(head[i]))
        tail_ids.append(F.asnumpy(tail[j]))
    scores = np.asarray(scores)
    scores = scores.reshape(scores.shape[0])
    head_ids = np.asarray(head_ids)
    tail_ids = np.asarray(tail_ids)
    idx = np.argsort(scores)
    if sfunc == 'cosine' or sfunc == 'dot' or sfunc == 'ext_jaccard':
        idx = idx[::-1]
    idx = idx[:10]
    head_ids = head_ids[idx]
    tail_ids = tail_ids[idx]
    score_topk = scores[idx]

    r1_head, r1_tail, r1_score = result1[0]
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)
    print('pass pair wise')

    head = F.arange(0, num_head)
    tail = F.arange(num_head, num_head+num_tail)
    result1 = sim_infer.topK(head, tail)
    assert len(result1) == 1
    scores = []
    head_ids = []
    tail_ids = []
    for i in range(head.shape[0]):
        for j in range(tail.shape[0]):
            hemb = F.take(emb, head[i], 0)
            temb = F.take(emb, tail[j], 0)

            score = sim_func(hemb, temb)
            scores.append(F.asnumpy(score))
            head_ids.append(F.asnumpy(head[i]))
            tail_ids.append(F.asnumpy(tail[j]))
    scores = np.asarray(scores)
    scores = scores.reshape(scores.shape[0])
    head_ids = np.asarray(head_ids)
    tail_ids = np.asarray(tail_ids)
    idx = np.argsort(scores)
    if sfunc == 'cosine' or sfunc == 'dot' or sfunc == 'ext_jaccard':
        idx = idx[::-1]
    idx = idx[:10]
    head_ids = head_ids[idx]
    tail_ids = tail_ids[idx]
    score_topk = scores[idx]

    r1_head, r1_tail, r1_score = result1[0]
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)

    emb_ids = F.arange(0, num_emb)
    result1 = sim_infer.topK(emb_ids, emb_ids, bcast=True)
    result2 = sim_infer.topK(bcast=True)
    assert len(result1) == emb_ids.shape[0]
    assert len(result2) == emb_ids.shape[0]

    for i in range(emb_ids.shape[0]):
        scores = []
        head_ids = []
        tail_ids = []
        for j in range(emb_ids.shape[0]):
            hemb = F.take(emb, emb_ids[i], 0)
            temb = F.take(emb, emb_ids[j], 0)
    
            score = sim_func(hemb, temb)
            score = F.asnumpy(score)
            scores.append(score)
            tail_ids.append(F.asnumpy(emb_ids[j]))
        scores = np.asarray(scores)
        scores = scores.reshape(scores.shape[0])
        tail_ids = np.asarray(tail_ids)
        idx = np.argsort(scores)
        if sfunc == 'cosine' or sfunc == 'dot' or sfunc == 'ext_jaccard':
            idx = idx[::-1]
        idx = idx[:10]
        head_ids = np.full((10,), F.asnumpy(emb_ids[i]))
        tail_ids = tail_ids[idx]
        score_topk = scores[idx]

        r1_head, r1_tail, r1_score = result1[i]
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        r2_head, r2_tail, r2_score = result2[i]
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)
    print('pass all')

def test_cosine_topk_emb():
    def cosine_func(x, y):
        score = F.sum(x * y, dim=0)
        x_norm2 = F.sum(x * x, dim=0) ** (1/2)
        y_norm2 = F.sum(y * y, dim=0) ** (1/2)

        return score / (x_norm2 * y_norm2)

    run_topk_emb('cosine', cosine_func)

def test_l2_topk_emb():
    def l2_func(x, y):
        score = x - y

        return F.sum(score * score, dim=0) ** (1/2)
    run_topk_emb('l2', l2_func)

def test_l1_topk_emb():
    def l1_func(x, y):
        score = x - y

        return norm(score, p=1)
    run_topk_emb('l1', l1_func)

def test_dot_topk_emb():
    def dot_func(x, y):
        return F.sum(x * y, dim=0)

    run_topk_emb('dot', dot_func)

def test_extended_jaccard_topk_emb():
    def extended_jaccard_func(x, y):
        score = F.sum(x * y, dim=0)
        x = F.sum(x * x, dim=0)
        y = F.sum(y * y, dim=0)

        return score / (x + y - score)
    run_topk_emb('ext_jaccard', extended_jaccard_func)

if __name__ == '__main__':
    test_topk_transe()
    test_topk_distmult()
    test_topk_complex()
    test_topk_rescal()
    #test_topk_transr()
    test_topk_rotate()
    test_cosine_topk_emb()
    test_l2_topk_emb()
    test_l1_topk_emb()
    test_dot_topk_emb()
    test_extended_jaccard_topk_emb()