# -*- coding: utf-8 -*-
#
# test_topk.py
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
from pathlib import Path
import tempfile

import scipy as sp
import dgl
import numpy as np
import dgl.backend as F
import dgl

from models import InferModel
from models.infer import ScoreInfer, EmbSimInfer
from models import TransEModel, TransE_l2Model, TransE_l1Model, TransRModel, RotatEModel
from models import DistMultModel, ComplExModel, RESCALModel
from models import GNNModel

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

def create_score_infer(model_name, entity_emb, rel_emb):
    hidden_dim = entity_emb.shape[1]
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
    elif model_name == 'SimplE':
        model = InferModel('cpu', model_name, hidden_dim, double_entity_emb=True, double_relation_emb=True)

    model.entity_emb = InferEmbedding('cpu')
    model.entity_emb.emb = entity_emb
    model.relation_emb = InferEmbedding('cpu')
    model.relation_emb.emb = rel_emb
    score_model.model = model
    score_func = model.score_func

    return score_model, score_func

def check_topk_score(model_name):
    hidden_dim = 32
    gamma = 12.0

    num_entity = 40
    num_rels = 4
    entity_emb, rel_emb = generate_rand_emb(model_name, num_entity, num_rels, hidden_dim, 'none')
    score_model, score_func = create_score_infer(model_name, entity_emb, rel_emb)

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

def create_kge_emb_sim(emb, sfunc):
    sim_infer = EmbSimInfer(-1, None, sfunc, 32)
    sim_infer.emb = emb
    return sim_infer

def test_topk_simple():
    check_topk_score('SimplE')

def run_topk_emb(sfunc, sim_func, create_emb_sim=create_kge_emb_sim):
    hidden_dim = 32
    num_head = 40
    num_tail = 40
    num_emb = 80

    emb = F.uniform((num_emb, hidden_dim), F.float32, F.cpu(), -1, 1)
    head = F.arange(0, num_head)
    tail = F.arange(num_head, num_head+num_tail)
    sim_infer = create_emb_sim(emb, sfunc)

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
    idx = idx[::-1]
    idx = idx[:10]
    head_ids = head_ids[idx]
    tail_ids = tail_ids[idx]
    score_topk = scores[idx]

    r1_head, r1_tail, r1_score = result1[0]
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)
    print('pass pairwise')

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

def test_cosine_topk_emb(create_emb_sim=create_kge_emb_sim):
    def cosine_func(x, y):
        score = F.sum(x * y, dim=0)
        x_norm2 = F.sum(x * x, dim=0) ** (1/2)
        y_norm2 = F.sum(y * y, dim=0) ** (1/2)

        return score / (x_norm2 * y_norm2)

    run_topk_emb('cosine', cosine_func, create_emb_sim=create_emb_sim)

def test_l2_topk_emb(create_emb_sim=create_kge_emb_sim):
    def l2_func(x, y):
        score = x - y

        return -F.sum(score * score, dim=0) ** (1/2)
    run_topk_emb('l2', l2_func, create_emb_sim=create_emb_sim)

def test_l1_topk_emb(create_emb_sim=create_kge_emb_sim):
    def l1_func(x, y):
        score = x - y

        return -norm(score, p=1)
    run_topk_emb('l1', l1_func, create_emb_sim=create_emb_sim)

def test_dot_topk_emb(create_emb_sim=create_kge_emb_sim):
    def dot_func(x, y):
        return F.sum(x * y, dim=0)

    run_topk_emb('dot', dot_func)

def test_extended_jaccard_topk_emb(create_emb_sim=create_kge_emb_sim):
    def extended_jaccard_func(x, y):
        score = F.sum(x * y, dim=0)
        x = F.sum(x * x, dim=0)
        y = F.sum(y * y, dim=0)

        return score / (x + y - score)
    run_topk_emb('ext_jaccard', extended_jaccard_func, create_emb_sim=create_emb_sim)

############################# BasicGEModel ##############################

def create_emb_file(path, filename, emb):
    np.save(os.path.join(path, filename), emb)

def check_topk_score2(score_model, exclude_mode):
    num_entity = 40
    num_rels = 4

    src = F.arange(0, num_entity)
    dst1 = src + 1
    dst1[num_entity-1] = 0
    dst2 = src - 1
    dst2[0] = num_entity-1
    src = F.cat([src, src], dim=0)
    dst = F.cat([dst1, dst2], dim=0)
    src = F.cat([src, src, src, src], dim=0)
    dst = F.cat([dst, dst, dst, dst], dim=0)
    etype = F.cat([th.full((num_entity*2,), 0, dtype=th.long),
                    th.full((num_entity*2,), 1, dtype=th.long),
                    th.full((num_entity*2,), 2, dtype=th.long),
                    th.full((num_entity*2,), 3, dtype=th.long)],
                    dim=0)
    g = dgl._deprecate.graph.DGLGraph((src, dst))
    g.edata['tid'] = etype

    _check_topk_score2(score_model, g, num_entity, num_rels, exclude_mode)

def _check_topk_score2(score_model, g, num_entity, num_rels, exclude_mode):
    hidden_dim = 32
    num_entity = 40
    num_rels = 4
    with tempfile.TemporaryDirectory() as tmpdirname:
        entity_emb, rel_emb = generate_rand_emb(score_model.model_name, num_entity, num_rels, hidden_dim, 'none')
        create_emb_file(Path(tmpdirname), 'entity.npy', entity_emb.numpy())
        create_emb_file(Path(tmpdirname), 'relation.npy', rel_emb.numpy())

        score_model.load(Path(tmpdirname))
        score_model.attach_graph(g)
        score_func = score_model._score_func

    head = F.arange(0, num_entity // 2)
    rel = F.arange(0, num_rels)
    tail = F.arange(num_entity // 2, num_entity)

    # exec_model==triplet_wise
    tw_rel = np.random.randint(0, num_rels, num_entity // 2)
    tw_rel = F.tensor(tw_rel)
    result1 = score_model.link_predict(head, tw_rel, tail, exec_mode='triplet_wise', exclude_mode=exclude_mode, batch_size=16)
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
    if exclude_mode is None or exclude_mode == 'mask':
        idx = idx[:10]
        head_ids = head_ids[idx]
        rel_ids = rel_ids[idx]
        tail_ids = tail_ids[idx]
        score_topk = scores[idx]
        if exclude_mode == 'mask':
            mask = np.zeros((10,))
            for i in range(10):
                if (head_ids[i] + 1) % num_entity == tail_ids[i] or \
                    (head_ids[i] - 1) % num_entity == tail_ids[i]:
                    mask[i] = 1
    else:
        c_head_idx = []
        c_rel_idx = []
        c_tail_idx = []
        c_score_topk = []
        cur_idx = 0
        while len(c_head_idx) < 10:
            c_idx = idx[cur_idx]
            cur_idx += 1
            if (head_ids[c_idx] + 1) % num_entity == tail_ids[c_idx] or \
                (head_ids[c_idx] - 1) % num_entity == tail_ids[c_idx]:
                continue
            c_head_idx.append(head_ids[c_idx])
            c_tail_idx.append(tail_ids[c_idx])
            c_rel_idx.append(rel_ids[c_idx])
            c_score_topk.append(scores[c_idx])
        head_ids = F.tensor(c_head_idx)
        rel_ids = F.tensor(c_rel_idx)
        tail_ids = F.tensor(c_tail_idx)
        score_topk = F.tensor(c_score_topk)

    r1_head, r1_rel, r1_tail, r1_score, r1_mask = result1[0]
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r1_rel, rel_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    if exclude_mode == 'mask':
        np.testing.assert_allclose(r1_mask, mask)
    else:
        assert r1_mask is None

    # exec_mode==all
    result1 = score_model.link_predict(head, rel, tail, topk=20, exclude_mode=exclude_mode, batch_size=16)
    result2 = score_model.link_predict(head=head, tail=tail, topk=20, exclude_mode=exclude_mode, batch_size=16)
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
    if exclude_mode is None or exclude_mode == 'mask':
        idx = idx[:20]
        head_ids = head_ids[idx]
        rel_ids = rel_ids[idx]
        tail_ids = tail_ids[idx]
        score_topk = scores[idx]
        if exclude_mode == 'mask':
            mask = np.zeros((20,))
            for i in range(20):
                if (head_ids[i] + 1) % num_entity == tail_ids[i] or \
                    (head_ids[i] - 1) % num_entity == tail_ids[i]:
                    mask[i] = 1
    else:
        c_head_idx = []
        c_rel_idx = []
        c_tail_idx = []
        c_score_topk = []
        cur_idx = 0
        while len(c_head_idx) < 20:
            c_idx = idx[cur_idx]
            cur_idx += 1
            if (head_ids[c_idx] + 1) % num_entity == tail_ids[c_idx] or \
                (head_ids[c_idx] - 1) % num_entity == tail_ids[c_idx]:
                continue
            c_head_idx.append(head_ids[c_idx])
            c_tail_idx.append(tail_ids[c_idx])
            c_rel_idx.append(rel_ids[c_idx])
            c_score_topk.append(scores[c_idx])
        head_ids = F.tensor(c_head_idx)
        rel_ids = F.tensor(c_rel_idx)
        tail_ids = F.tensor(c_tail_idx)
        score_topk = F.tensor(c_score_topk)

    r1_head, r1_rel, r1_tail, r1_score, r1_mask = result1[0]
    r2_head, r2_rel, r2_tail, r2_score, r2_mask = result2[0]
    np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1_head, head_ids)
    np.testing.assert_allclose(r2_head, head_ids)
    np.testing.assert_allclose(r1_rel, rel_ids)
    np.testing.assert_allclose(r2_rel, rel_ids)
    np.testing.assert_allclose(r1_tail, tail_ids)
    np.testing.assert_allclose(r2_tail, tail_ids)
    if exclude_mode == 'mask':
        np.testing.assert_allclose(r1_mask, mask)
        np.testing.assert_allclose(r2_mask, mask)
    else:
        assert r1_mask is None
        assert r2_mask is None

    result1 = score_model.link_predict(head, rel, tail, exec_mode='batch_rel', exclude_mode=exclude_mode, batch_size=16)
    result2 = score_model.link_predict(head=head, tail=tail, exec_mode='batch_rel', exclude_mode=exclude_mode, batch_size=16)
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
        if exclude_mode is None or exclude_mode == 'mask':
            idx = idx[:10]
            head_ids = head_ids[idx]
            rel_ids = rel_ids[idx]
            tail_ids = tail_ids[idx]
            score_topk = scores[idx]
            if exclude_mode == 'mask':
                mask = np.full((10,), False)
                for i in range(10):
                    if (head_ids[i] + 1) % num_entity == tail_ids[i] or \
                        (head_ids[i] - 1) % num_entity == tail_ids[i]:
                        mask[i] = True
        else:
            c_head_idx = []
            c_rel_idx = []
            c_tail_idx = []
            c_score_topk = []
            cur_idx = 0
            while len(c_head_idx) < 10:
                c_idx = idx[cur_idx]
                cur_idx += 1
                if (head_ids[c_idx] + 1) % num_entity == tail_ids[c_idx] or \
                    (head_ids[c_idx] - 1) % num_entity == tail_ids[c_idx]:
                    continue
                c_head_idx.append(head_ids[c_idx])
                c_tail_idx.append(tail_ids[c_idx])
                c_rel_idx.append(rel_ids[c_idx])
                c_score_topk.append(scores[c_idx])
            head_ids = F.tensor(c_head_idx)
            rel_ids = F.tensor(c_rel_idx)
            tail_ids = F.tensor(c_tail_idx)
            score_topk = F.tensor(c_score_topk)

        r1_head, r1_rel, r1_tail, r1_score, r1_mask = result1[j]
        r2_head, r2_rel, r2_tail, r2_score, r2_mask = result2[j]
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r1_rel, rel_ids)
        np.testing.assert_allclose(r2_rel, rel_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)
        if exclude_mode == 'mask':
            np.testing.assert_allclose(r1_mask, mask)
            np.testing.assert_allclose(r2_mask, mask)
        else:
            assert r1_mask is None
            assert r2_mask is None


    head = F.arange(0, num_entity)
    rel = F.arange(0, num_rels)
    tail = F.arange(0, num_entity)
    result1 = score_model.link_predict(head, rel, tail, exec_mode='batch_head', exclude_mode=exclude_mode, batch_size=16)
    result2 = score_model.link_predict(exec_mode='batch_head', exclude_mode=exclude_mode, batch_size=16)
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
        if exclude_mode is None or exclude_mode == 'mask':
            idx = idx[:10]
            head_ids = head_ids[idx]
            rel_ids = rel_ids[idx]
            tail_ids = tail_ids[idx]
            score_topk = scores[idx]
            if exclude_mode == 'mask':
                mask = np.full((10,), False)
                for l in range(10):
                    if (head_ids[l] + 1) % num_entity == tail_ids[l] or \
                        (head_ids[l] - 1) % num_entity == tail_ids[l]:
                        mask[l] = True
        else:
            c_head_idx = []
            c_rel_idx = []
            c_tail_idx = []
            c_score_topk = []
            cur_idx = 0
            while len(c_head_idx) < 10:
                c_idx = idx[cur_idx]
                cur_idx += 1
                if (head_ids[c_idx] + 1) % num_entity == tail_ids[c_idx] or \
                    (head_ids[c_idx] - 1) % num_entity == tail_ids[c_idx]:
                    continue
                c_head_idx.append(head_ids[c_idx])
                c_tail_idx.append(tail_ids[c_idx])
                c_rel_idx.append(rel_ids[c_idx])
                c_score_topk.append(scores[c_idx])
            head_ids = F.tensor(c_head_idx)
            rel_ids = F.tensor(c_rel_idx)
            tail_ids = F.tensor(c_tail_idx)
            score_topk = F.tensor(c_score_topk)

        r1_head, r1_rel, r1_tail, r1_score, r1_mask = result1[i]
        r2_head, r2_rel, r2_tail, r2_score, r2_mask = result2[i]
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r1_rel, rel_ids)
        np.testing.assert_allclose(r2_rel, rel_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
        if exclude_mode == 'mask':
            np.testing.assert_allclose(r1_mask, mask)
            np.testing.assert_allclose(r2_mask, mask)
        else:
            assert r1_mask is None
            assert r2_mask is None

    result1 = score_model.link_predict(head, rel, tail, exec_mode='batch_tail', exclude_mode=exclude_mode)
    result2 = score_model.link_predict(exec_mode='batch_tail', exclude_mode=exclude_mode)
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
        if exclude_mode is None or exclude_mode == 'mask':
            idx = idx[:10]
            head_ids = head_ids[idx]
            rel_ids = rel_ids[idx]
            tail_ids = tail_ids[idx]
            score_topk = scores[idx]
            if exclude_mode == 'mask':
                mask = np.full((10,), False)
                for l in range(10):
                    if (head_ids[l] + 1) % num_entity == tail_ids[l] or \
                        (head_ids[l] - 1) % num_entity == tail_ids[l]:
                        mask[l] = True
        else:
            c_head_idx = []
            c_rel_idx = []
            c_tail_idx = []
            c_score_topk = []
            cur_idx = 0
            while len(c_head_idx) < 10:
                c_idx = idx[cur_idx]
                cur_idx += 1
                if (head_ids[c_idx] + 1) % num_entity == tail_ids[c_idx] or \
                    (head_ids[c_idx] - 1) % num_entity == tail_ids[c_idx]:
                    continue
                c_head_idx.append(head_ids[c_idx])
                c_tail_idx.append(tail_ids[c_idx])
                c_rel_idx.append(rel_ids[c_idx])
                c_score_topk.append(scores[c_idx])
            head_ids = F.tensor(c_head_idx)
            rel_ids = F.tensor(c_rel_idx)
            tail_ids = F.tensor(c_tail_idx)
            score_topk = F.tensor(c_score_topk)

        r1_head, r1_rel, r1_tail, r1_score, r1_mask = result1[k]
        r2_head, r2_rel, r2_tail, r2_score, r2_mask = result2[k]
        np.testing.assert_allclose(r1_head, head_ids)
        np.testing.assert_allclose(r2_head, head_ids)
        np.testing.assert_allclose(r1_rel, rel_ids)
        np.testing.assert_allclose(r2_rel, rel_ids)
        np.testing.assert_allclose(r1_tail, tail_ids)
        np.testing.assert_allclose(r2_tail, tail_ids)
        np.testing.assert_allclose(r1_score, score_topk, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(r2_score, score_topk, rtol=1e-5, atol=1e-5)
        if exclude_mode == 'mask':
            np.testing.assert_allclose(r1_mask, mask)
            np.testing.assert_allclose(r2_mask, mask)
        else:
            assert r1_mask is None
            assert r2_mask is None

def test_transe_model_topk(device='cpu'):
    gamma = 12.0
    transe_model = TransEModel(device, gamma)
    check_topk_score2(transe_model, exclude_mode=None)
    check_topk_score2(transe_model, exclude_mode='mask')
    check_topk_score2(transe_model, exclude_mode='exclude')
    transe_model = TransE_l2Model(device, gamma)
    check_topk_score2(transe_model, exclude_mode=None)
    check_topk_score2(transe_model, exclude_mode='mask')
    check_topk_score2(transe_model, exclude_mode='exclude')
    transe_model = TransE_l1Model(device, gamma)
    check_topk_score2(transe_model, exclude_mode=None)
    check_topk_score2(transe_model, exclude_mode='mask')
    check_topk_score2(transe_model, exclude_mode='exclude')

def test_distmult_model_topk(device='cpu'):
    model = DistMultModel(device)
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')

def test_complex_model_topk(device='cpu'):
    model = ComplExModel(device)
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')

def test_rescal_model_topk(device='cpu'):
    model = RESCALModel(device)
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')

def test_rotate_model_topk(device='cpu'):
    gamma = 12.0
    model = RotatEModel(device, gamma)
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')

def test_gnn_model_topk(device='cpu'):
    gamma = 12.0
    model = GNNModel(device, 'TransE', gamma)
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')
    model = GNNModel(device, 'TransE_l1', gamma)
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')
    model = GNNModel(device, 'DistMult')
    check_topk_score2(model, exclude_mode=None)
    check_topk_score2(model, exclude_mode='mask')
    check_topk_score2(model, exclude_mode='exclude')

def run_topk_emb2(sfunc, sim_func, emb_model):
    hidden_dim = 32
    num_head = 40
    num_tail = 40
    num_emb = 80

    with tempfile.TemporaryDirectory() as tmpdirname:
        emb = F.uniform((num_emb, hidden_dim), F.float32, F.cpu(), -1, 1)
        create_emb_file(Path(tmpdirname), 'entity.npy', emb.numpy())
        create_emb_file(Path(tmpdirname), 'relation.npy', emb.numpy())

        emb_model.load(Path(tmpdirname))

    head = F.arange(0, num_head)
    tail = F.arange(num_head, num_head+num_tail)
    result1 = emb_model.embed_sim(head, tail, 'entity', sfunc=sfunc, pair_ws=True)
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
    result1 = emb_model.embed_sim(head, tail, 'entity', sfunc=sfunc)
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
    result1 = emb_model.embed_sim(emb_ids, emb_ids, 'entity', sfunc=sfunc, bcast=True)
    result2 = emb_model.embed_sim(embed_type='entity', sfunc=sfunc, bcast=True)
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

def _test_cosine_topk_emb2(emb_model):
    def cosine_func(x, y):
        score = F.sum(x * y, dim=0)
        x_norm2 = F.sum(x * x, dim=0) ** (1/2)
        y_norm2 = F.sum(y * y, dim=0) ** (1/2)

        return score / (x_norm2 * y_norm2)

    run_topk_emb2('cosine', cosine_func, emb_model=emb_model)

def _test_l2_topk_emb2(emb_model):
    def l2_func(x, y):
        score = x - y

        return -F.sum(score * score, dim=0) ** (1/2)
    run_topk_emb2('l2', l2_func, emb_model=emb_model)

def _test_l1_topk_emb2(emb_model):
    def l1_func(x, y):
        score = x - y

        return -norm(score, p=1)
    run_topk_emb2('l1', l1_func, emb_model=emb_model)

def _test_dot_topk_emb2(emb_model):
    def dot_func(x, y):
        return F.sum(x * y, dim=0)

    run_topk_emb2('dot', dot_func, emb_model=emb_model)

def _test_extended_jaccard_topk_emb2(emb_model):
    def extended_jaccard_func(x, y):
        score = F.sum(x * y, dim=0)
        x = F.sum(x * x, dim=0)
        y = F.sum(y * y, dim=0)

        return score / (x + y - score)
    run_topk_emb2('ext_jaccard', extended_jaccard_func, emb_model=emb_model)


def test_transe_model_topk_emb(device='cpu'):
    gamma = 12.0
    transe_model = TransEModel(device, gamma)
    _test_cosine_topk_emb2(transe_model)
    transe_model = TransE_l2Model(device, gamma)
    _test_cosine_topk_emb2(transe_model)
    transe_model = TransE_l1Model(device, gamma)
    _test_cosine_topk_emb2(transe_model)

def test_distmult_model_topk_emb(device='cpu'):
    model = DistMultModel(device)
    _test_l2_topk_emb2(model)

def test_complex_model_topk_emb(device='cpu'):
    model = ComplExModel(device)
    _test_l1_topk_emb2(model)

def test_rescal_model_topk_emb(device='cpu'):
    model = RESCALModel(device)
    _test_dot_topk_emb2(model)

def test_rotate_model_topk_emb(device='cpu'):
    gamma = 12.0
    model = RotatEModel(device, gamma)
    _test_extended_jaccard_topk_emb2(model)

def test_gnn_model_topk_emb(device='cpu'):
    gamma = 12.0
    model = GNNModel(device, 'TransE', gamma)
    _test_cosine_topk_emb2(model)
    model = GNNModel(device, 'TransE_l1', gamma)
    _test_l2_topk_emb2(model)
    model = GNNModel(device, 'DistMult')
    _test_l1_topk_emb2(model)

if __name__ == '__main__':
    test_topk_transe()
    test_topk_distmult()
    test_topk_complex()
    test_topk_rescal()
    #test_topk_transr()
    test_topk_simple()
    test_topk_rotate()
    test_cosine_topk_emb()
    test_l2_topk_emb()
    test_l1_topk_emb()
    test_dot_topk_emb()
    test_extended_jaccard_topk_emb()

    test_transe_model_topk()
    test_distmult_model_topk()
    test_complex_model_topk()
    test_rescal_model_topk()
    #test_transr_model_topk()
    test_rotate_model_topk()
    test_gnn_model_topk()

    test_transe_model_topk_emb()
    test_distmult_model_topk_emb()
    test_complex_model_topk_emb()
    test_rescal_model_topk_emb()
    #test_transr_model_topk_emb()
    test_rotate_model_topk_emb()
    test_gnn_model_topk_emb()

    test_transe_model_topk(device=0)
    test_distmult_model_topk(device=0)
    test_complex_model_topk(device=0)
    test_rescal_model_topk(device=0)
    #test_transr_model_topk()
    test_rotate_model_topk(device=0)
    test_gnn_model_topk(device=0)

    test_transe_model_topk_emb(device=0)
    test_distmult_model_topk_emb(device=0)
    test_complex_model_topk_emb(device=0)
    test_rescal_model_topk_emb(device=0)
    #test_transr_model_topk_emb()
    test_rotate_model_topk_emb(device=0)
    test_gnn_model_topk_emb(device=0)
