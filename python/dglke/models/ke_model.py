# -*- coding: utf-8 -*-
#
# ke_model.py
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
Knowledge Graph Embedding Model
1. TransE_1
2. TransE_2
3. TransR
4. RESCAL
5. DistMult
6. ComplEx
7. RotatE
8. SimplE
9. ConvE
"""
from abc import ABC, abstractmethod
from tqdm import trange, tqdm
import dgl

import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adagrad, Adam
from torch.utils.data import DataLoader

from .pytorch.tensor_models import logsigmoid
from .pytorch.tensor_models import none
from .pytorch.score_fun import *
from .pytorch.ke_tensor import KGEmbedding
from .pytorch.tensor_models import cosine_dist
from .pytorch.tensor_models import l2_dist
from .pytorch.tensor_models import l1_dist
from .pytorch.tensor_models import dot_dist
from .pytorch.tensor_models import extended_jaccard_dist
from .pytorch.tensor_models import floor_divide
from .pytorch.loss import LossGenerator
from .pytorch.regularizer import Regularizer
from dglke.util import thread_wrapped_func, Logger, get_compatible_batch_size, prepare_save_path, get_scalar, to_device
from dglke.dataloader import EvalDataset, TrainDataset, SequentialTotalSampler, PartitionChunkDataset
from dglke.dataloader import get_dataset

import time
import logging
import json

# debug package
from pyinstrument import Profiler

EMB_INIT_EPS = 2.0
PRECISION_EPS = 1e-5
DEFAULT_INFER_BATCHSIZE = 1024

print = Logger.print

class BasicGEModel(object):
    """ Basic Graph Embeding Model
    """

    def __init__(self, device, model_name, score_func):
        self._g = None
        self._model_name = model_name
        self._device = device
        self._entity_emb = KGEmbedding(device)
        self._relation_emb = KGEmbedding(device)
        self._score_func = score_func

    def attach_graph(self, g, etid_field='tid', ntid_field='ntid'):
        """ Attach dataset into Graph Embedding Model

        Parameter
        ----------
        g: DGLGraph
            Input data for knowledge graph
        etid_field: str
            Edge feature name storing the edge type id
        ntid_field: str
            Node feature name storing the node type id

        Note
        ----
        If the input graph is DGLGraph, we assume that it uses a homogeneous graph
        to represent the heterogeneous graph. The edge type id is stored in etid_field
        and the node type id is stored in ntid_field.
        """
        self._etid_field = etid_field
        self._ntid_field = ntid_field
        assert isinstance(g, dgl.DGLGraph)
        self._g = g

    def load(self, model_path):
        """ Load Graph Embedding Model from model_path.

        The default entity embeding file is entity.npy.
        The default relation embedding file is relation.npy.

        Parameter
        ---------
        model_path : str
            Path to store the model information
        """
        pass

    def save(self, model_path):
        """ Save Graph Embedding Model into model_path.

        All model related data are saved under model_path.
        The default entity embeding file is entity.npy.
        The default relation embedding file is relation.npy.

        Parameter
        ---------
        model_path : str
            Path to store the model information
        """
        assert False, 'Not support training now'

    def fit(self):
        """ Start training
        """
        assert False, 'Not support training now'

    def test(self):
        """ Start evaluation
        """
        assert False, 'Not support evaluation now'

    def _infer_score_func(self, head, rel, tail, triplet_wise=False, batch_size=DEFAULT_INFER_BATCHSIZE):
        head_emb = self.entity_embed[head]
        rel_emb = self.relation_embed[rel]
        tail_emb = self.entity_embed[tail]

        num_head = head.shape[0]
        num_rel = rel.shape[0]
        num_tail = tail.shape[0]

        score = []
        if triplet_wise:
            # triplet wise score: head, relation and tail tensor have the same length N,
            # for i in range(N):
            #     result.append(score(head[i],rel[i],tail[i]))
            class FakeEdge(object):
                def __init__(self, head_emb, rel_emb, tail_emb, device=-1):
                    self._hobj = {}
                    self._robj = {}
                    self._tobj = {}
                    self._hobj['emb'] = head_emb.to(device)
                    self._robj['emb'] = rel_emb.to(device)
                    self._tobj['emb'] = tail_emb.to(device)

                @property
                def src(self):
                    return self._hobj

                @property
                def dst(self):
                    return self._tobj

                @property
                def data(self):
                    return self._robj

            # calculate scores in mini-batches
            # so we can use GPU to accelerate the speed with avoiding GPU OOM
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sr_emb = rel_emb[i * batch_size : (i + 1) * batch_size \
                                                  if (i + 1) * batch_size < num_head \
                                                  else num_head]
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                edata = FakeEdge(sh_emb, sr_emb, st_emb, self._device)
                score.append(self._score_func.edge_func(edata)['score'].to(th.device('cpu')))
            score = th.cat(score, dim=0)
            return score
        else:
            # head, relation and tail tensors has different size
            # for h_i in range(head):
            #     for r_j in range(relation):
            #         for t_k in range(tail):
            #             result.append(score(h_i, r_j, t_k))
            # The result will have shape (len(head), len(relation), len(tail))
            rel_emb = rel_emb.to(self._device)

            # calculating scores using mini-batch, the default batchsize if 1024
            # This can avoid OOM when using GPU
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                s_score = []
                sh_emb = sh_emb.to(self._device)
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                       if (j + 1) * batch_size < num_tail \
                                                       else num_tail]
                    st_emb = st_emb.to(self._device)
                    s_score.append(self._score_func.infer(sh_emb, rel_emb, st_emb).to(th.device('cpu')))
                score.append(th.cat(s_score, dim=2))
            score = th.cat(score, dim=0)
            return th.reshape(score, (num_head, num_rel, num_tail))

    def _exclude_pos(self, sidx, score, idx, head, rel, tail, topk, exec_mode, exclude_mode):
        g = self._g
        num_triples = idx.shape[0]
        num_head = 1 if exec_mode == 'batch_head' else head.shape[0]
        num_rel = 1 if exec_mode == 'batch_rel' else rel.shape[0]
        num_tail = 1 if exec_mode == 'batch_tail' else tail.shape[0]

        res_head = []
        res_rel = []
        res_tail = []
        res_score = []
        result = []
        if exclude_mode == 'exclude':
            # exclude existing edges
            cur_k = 0
            batch_size = topk
            while (cur_k < num_triples):
                cur_sidx = sidx[cur_k:cur_k + batch_size if cur_k + batch_size < num_triples else num_triples]
                cur_score = score[cur_k:cur_k + batch_size if cur_k + batch_size < num_triples else num_triples]
                cur_idx = idx[cur_sidx]

                if exec_mode == 'triplet_wise':
                    cur_head = head[cur_idx]
                    cur_rel = rel[cur_idx]
                    cur_tail = tail[cur_idx]
                elif exec_mode == 'all':
                    tail_idx = cur_idx % num_tail
                    cur_idx = floor_divide(cur_idx, num_tail)
                    rel_idx = cur_idx % num_rel
                    cur_idx = floor_divide(cur_idx, num_rel)
                    head_idx = cur_idx % num_head

                    cur_head = head[head_idx]
                    cur_rel = rel[rel_idx]
                    cur_tail = tail[tail_idx]
                elif exec_mode == 'batch_head':
                    tail_idx = cur_idx % num_tail
                    cur_idx = floor_divide(cur_idx, num_tail)
                    rel_idx = cur_idx % num_rel

                    cur_head = th.full((cur_sidx.shape[0],), head, dtype=head.dtype)
                    cur_rel = rel[rel_idx]
                    cur_tail = tail[tail_idx]
                elif exec_mode == 'batch_rel':
                    tail_idx = cur_idx % num_tail
                    cur_idx = floor_divide(cur_idx, num_tail)
                    head_idx = cur_idx % num_head

                    cur_head = head[head_idx]
                    cur_rel = th.full((cur_sidx.shape[0],), rel, dtype=rel.dtype)
                    cur_tail = tail[tail_idx]
                elif exec_mode == 'batch_tail':
                    rel_idx = cur_idx % num_rel
                    cur_idx = floor_divide(cur_idx, num_rel)
                    head_idx = cur_idx % num_head

                    cur_head = head[head_idx]
                    cur_rel = rel[rel_idx]
                    cur_tail = th.full((cur_sidx.shape[0],), tail, dtype=tail.dtype)

                # Find exising edges
                # It is expacted that the existing edges are much less than triples
                # The idea is: 1) we get existing edges using g.edge_ids
                #              2) sort edges according to source node id (O(nlog(n)), n is number of edges)
                #              3) sort candidate triples according to cur_head (O(mlog(m)), m is number of cur_head nodes)
                #              4) go over all candidate triples and compare with existing edges,
                #                 as both edges and candidate triples are sorted. filtering edges out
                #                 will take only O(n+m)
                #              5) sort the score again it taks O(klog(k))
                uid, vid, eid = g.edge_ids(cur_head, cur_tail, return_uv=True)
                rid = g.edata[self._etid_field][eid]

                for i in range(cur_head.shape[0]):
                    h = cur_head[i]
                    r = cur_rel[i]
                    t = cur_tail[i]

                    h_where = uid == h
                    t_where = vid[h_where] == t
                    r_where = rid[h_where][t_where]
                    edge_exist = False
                    if r_where.shape[0] > 0:
                        for c_r in r_where:
                            if c_r == r:
                                edge_exist = True
                                break

                    if edge_exist is False:
                        res_head.append(h)
                        res_rel.append(r)
                        res_tail.append(t)
                        res_score.append(cur_score[i])

                if len(res_head) >= topk:
                    break

                cur_k += batch_size
                batch_size = topk - len(res_head) # check more edges
                batch_size = 16 if batch_size < 16 else batch_size # avoid tailing issue
            res_head = th.tensor(res_head)
            res_rel = th.tensor(res_rel)
            res_tail = th.tensor(res_tail)
            res_score = th.tensor(res_score)
            sidx = th.argsort(res_score, dim=0, descending=True)
            sidx = sidx[:topk] if topk < sidx.shape[0] else sidx
            result.append((res_head[sidx],
                           res_rel[sidx],
                           res_tail[sidx],
                           res_score[sidx],
                           None))
        else:
            # including the existing edges in the result
            topk = topk if topk < num_triples else num_triples
            sidx = sidx[:topk]
            idx = idx[sidx]

            if exec_mode == 'triplet_wise':
                head = head[idx]
                rel = rel[idx]
                tail = tail[idx]
            elif exec_mode == 'all':
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                rel_idx = idx % num_rel
                idx = floor_divide(idx, num_rel)
                head_idx = idx % num_head

                head = head[head_idx]
                rel = rel[rel_idx]
                tail = tail[tail_idx]
            elif exec_mode == 'batch_head':
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                rel_idx = idx % num_rel

                head = th.full((topk,), head, dtype=head.dtype)
                rel = rel[rel_idx]
                tail = tail[tail_idx]
            elif exec_mode == 'batch_rel':
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                head = head[head_idx]
                rel = th.full((topk,), rel, dtype=rel.dtype)
                tail = tail[tail_idx]
            elif exec_mode == 'batch_tail':
                rel_idx = idx % num_rel
                idx = floor_divide(idx, num_rel)
                head_idx = idx % num_head

                head = head[head_idx]
                rel = rel[rel_idx]
                tail = th.full((topk,), tail, dtype=tail.dtype)

            if exclude_mode == 'mask':
                # Find exising edges
                # It is expacted that the existing edges are much less than triples
                # The idea is: 1) we get existing edges using g.edge_ids
                #              2) sort edges according to source node id (O(nlog(n)), n is number of edges)
                #              3) sort candidate triples according to cur_head (O(mlog(m)), m is number of cur_head nodes)
                #              4) go over all candidate triples and compare with existing edges and mask them,
                #                 as both edges and candidate triples are sorted. filtering edges out
                #                 will take only O(n+m)
                uid, vid, eid = g.edge_ids(head, tail, return_uv=True)
                rid = g.edata[self._etid_field][eid]
                mask = th.full((head.shape[0],), False, dtype=th.bool)

                if len(uid) > 0:
                    for i in range(head.shape[0]):
                        h = head[i]
                        r = rel[i]
                        t = tail[i]

                        h_where = uid == h
                        t_where = vid[h_where] == t
                        r_where = rid[h_where][t_where]
                        if r_where.shape[0] > 0:
                            for c_r in r_where:
                                if c_r == r:
                                    mask[i] = True
                                    break

                result.append((head, rel, tail, score, mask))
            else:
                result.append((head, rel, tail, score, None))

        return result

    def _topk_exclude_pos(self, score, idx, head, rel, tail, topk, exec_mode, exclude_mode):
        """ Generate topk most relevent triplets and corresponding scores.

            It takes following steps:

              1) find topk elements
              2) sort topk elements in descending order
              3) call _exclude_pos if figure out existing edges
        """
        if exclude_mode == 'exclude':
            if idx.shape[0] < topk * 4: # TODO(xiangsx): Find a better value of topk * n
                topk_score, topk_sidx = th.topk(score, k=idx.shape[0], dim=0)
                sidx = th.argsort(topk_score, dim=0, descending=True)
                sidx = topk_sidx[sidx]
                result = self._exclude_pos(sidx=sidx,
                                           score=topk_score,
                                           idx=idx,
                                           head=head,
                                           rel=rel,
                                           tail=tail,
                                           topk=topk,
                                           exec_mode=exec_mode,
                                           exclude_mode=exclude_mode)
            else:
                topk_score, topk_sidx = th.topk(score, k= topk * 4, dim=0)
                sidx = th.argsort(topk_score, dim=0, descending=True)
                sidx = topk_sidx[sidx]
                result = self._exclude_pos(sidx=sidx,
                                           score=topk_score,
                                           idx=idx,
                                           head=head,
                                           rel=rel,
                                           tail=tail,
                                           topk=topk,
                                           exec_mode=exec_mode,
                                           exclude_mode=exclude_mode)
                if len(result) < topk:
                    sidx = th.argsort(score, dim=0, descending=True)
                    result = self._exclude_pos(sidx=sidx,
                                               score=score[sidx],
                                               idx=idx,
                                               head=head,
                                               rel=rel,
                                               tail=tail,
                                               topk=topk,
                                               exec_mode=exec_mode,
                                               exclude_mode=exclude_mode)
        else:
            topk = idx.shape[0] if idx.shape[0] < topk else topk
            topk_score, topk_sidx = th.topk(score, k=topk, dim=0)
            sidx = th.argsort(topk_score, dim=0, descending=True)
            sidx = topk_sidx[sidx]
            result = self._exclude_pos(sidx=sidx,
                                       score=topk_score,
                                       idx=idx,
                                       head=head,
                                       rel=rel,
                                       tail=tail,
                                       topk=topk,
                                       exec_mode=exec_mode,
                                       exclude_mode=exclude_mode)
        return result

    def link_predict(self, head=None, rel=None, tail=None, exec_mode='all', sfunc='none', topk=10, exclude_mode=None, batch_size=DEFAULT_INFER_BATCHSIZE):
        """ Predicts missing entities or relations in a triplet.

        Given head_id, relation_id and tail_id, return topk most relevent triplet.

        Parameters
        ----------
        head: th.Tensor
            A tensor of head entity id.

        rel: th.Tensor
            A tensor of relation id.

        tail: th.Tensor
            A tensor of tail entity id.

        exec_mode: str
            How to calculate scores for triplets and calculate topK:

              * triplet_wise: head, relation and tail lists have the same length N,
                and we calculate the similarity triplet by triplet:
                ``result = topK([score(h_i, r_i, t_i) for i in N])``,
                the result shape will be (K,)

              * all: three lists of head, relation and tail ids are provided as H, R and T,
                and we calculate all possible combinations of all triplets (h_i, r_j, t_k):
                ``result = topK([[[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R] for each t_k in T])``,
                the result shape will be (K,)

              * batch_head: three lists of head, relation and tail ids are provided as H, R and T
                and we calculate topK for each element in head:
                ``result = topK([[score(h_i, r_j, t_k) for each r_j in R] for each t_k in T]) for each h_i in H``
                the result shape will be (sizeof(H), K)

              * batch_rel: three lists of head, relation and tail ids are provided as H, R and T,
                and we calculate topK for each element in relation:
                ``result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each t_k in T]) for each r_j in R``,
                the result shape will be (sizeof(R), K)

              * batch_tail: three lists of head, relation and tail ids are provided as H, R and T,
                and we calculate topK for each element in tail:
                ``result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R]) for each t_k in T``,
                the result shape will be (sizeof(T), K)

        sfunc: str
            What kind of score is used in ranking and will be output:

              * none: $score = x$
              * logsigmoid: $score = log(sigmoid(x))

        topk: int
            Return top k results

        exclude_mode: str
            Whether to exclude positive edges:

            * None: Do not exclude positive edges.

            * 'mask': Return topk edges and a mask indicating which one is positive edge.

            * 'exclude': Exclude positive edges, the returned k edges will be missing edges in the graph.

        Return
        ------
        A list of (head_idx, rel_idx, tail_idx, score)
        """
        if head is None:
            head = th.arange(0, self.num_entity)
        else:
            head = th.tensor(head)
        if rel is None:
            rel = th.arange(0, self.num_rel)
        else:
            rel = th.tensor(rel)
        if tail is None:
            tail = th.arange(0, self.num_entity)
        else:
            tail = th.tensor(tail)

        num_head = head.shape[0]
        num_rel = rel.shape[0]
        num_tail = tail.shape[0]

        if sfunc == 'none':
            sfunc = none
        else:
            sfunc = logsigmoid

        # if exclude_mode is not None, we need a graph to do the edge filtering
        assert (self._g is not None) or (exclude_mode is None), \
            'If exclude_mode is not None, please use load_graph() to initialize ' \
            'a graph for edge filtering.'
        if exec_mode == 'triplet_wise':
            assert num_head == num_rel, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'
            assert num_head == num_tail, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'

            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail, triplet_wise=True, batch_size=batch_size)
                score = sfunc(raw_score)
                idx = th.arange(0, num_head)

            result = self._topk_exclude_pos(score=score,
                                            idx=idx,
                                            head=head,
                                            rel=rel,
                                            tail=tail,
                                            topk=topk,
                                            exec_mode=exec_mode,
                                            exclude_mode=exclude_mode)
        elif exec_mode == 'all':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
                raw_score = th.reshape(raw_score, (head.shape[0]*rel.shape[0]*tail.shape[0],))
                score = sfunc(raw_score)
            idx = th.arange(0, num_head * num_rel * num_tail)

            result = self._topk_exclude_pos(score=score,
                                            idx=idx,
                                            head=head,
                                            rel=rel,
                                            tail=tail,
                                            topk=topk,
                                            exec_mode=exec_mode,
                                            exclude_mode=exclude_mode)
        elif exec_mode == 'batch_head':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
            for i in range(num_head):
                score = sfunc(th.reshape(raw_score[i,:,:], (rel.shape[0]*tail.shape[0],)))
                idx = th.arange(0, num_rel * num_tail)

                res = self._topk_exclude_pos(score=score,
                                             idx=idx,
                                             head=head[i],
                                             rel=rel,
                                             tail=tail,
                                             topk=topk,
                                             exec_mode=exec_mode,
                                             exclude_mode=exclude_mode)

                result.append(res[0])
        elif exec_mode == 'batch_rel':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
            for i in range(num_rel):
                score = sfunc(th.reshape(raw_score[:,i,:], (head.shape[0]*tail.shape[0],)))
                idx = th.arange(0, num_head * num_tail)

                res = self._topk_exclude_pos(score=score,
                                             idx=idx,
                                             head=head,
                                             rel=rel[i],
                                             tail=tail,
                                             topk=topk,
                                             exec_mode=exec_mode,
                                             exclude_mode=exclude_mode)

                result.append(res[0])
        elif exec_mode == 'batch_tail':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
            for i in range(num_tail):
                score = sfunc(th.reshape(raw_score[:,:,i], (head.shape[0]*rel.shape[0],)))
                idx = th.arange(0, num_head * num_rel)

                res = self._topk_exclude_pos(score=score,
                                             idx=idx,
                                             head=head,
                                             rel=rel,
                                             tail=tail[i],
                                             topk=topk,
                                             exec_mode=exec_mode,
                                             exclude_mode=exclude_mode)

                result.append(res[0])
        else:
            assert False, 'unknow execution mode type {}'.format(exec_mode)

        return result

    def _embed_sim(self, head, tail, emb, sfunc='cosine', bcast=False, pair_ws=False, topk=10):
        batch_size=DEFAULT_INFER_BATCHSIZE
        if head is None:
            head = th.arange(0, emb.shape[0])
        else:
            head = th.tensor(head)
        if tail is None:
            tail = th.arange(0, emb.shape[0])
        else:
            tail = th.tensor(tail)
        head_emb = emb[head]
        tail_emb = emb[tail]

        if sfunc == 'cosine':
            sim_func = cosine_dist
        elif sfunc == 'l2':
            sim_func = l2_dist
        elif sfunc == 'l1':
            sim_func = l1_dist
        elif sfunc == 'dot':
            sim_func = dot_dist
        elif sfunc == 'ext_jaccard':
            sim_func = extended_jaccard_dist

        if pair_ws is True:
            result = []
            # chunked cal score
            score = []
            num_head = head.shape[0]
            num_tail = tail.shape[0]

            # calculating scores using mini-batch, the default batchsize if 1024
            # This can avoid OOM when using GPU
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sh_emb = sh_emb.to(self._device)
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                st_emb = st_emb.to(self._device)
                score.append(sim_func(sh_emb, st_emb, pw=True).to(th.device('cpu')))
            score = th.cat(score, dim=0)

            topk_score, topk_sidx = th.topk(score,
                                            k=topk if score.shape[0] > topk else score.shape[0],
                                            dim=0)
            sidx = th.argsort(topk_score, dim=0, descending=True)
            sidx = topk_sidx[sidx]
            score = score[sidx]
            result.append((head[sidx],
                           tail[sidx],
                           score))
        else:
            num_head = head.shape[0]
            num_tail = tail.shape[0]

            # calculating scores using mini-batch, the default batchsize if 1024
            # This can avoid OOM when using GPU
            score = []
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                            if (i + 1) * batch_size < num_head \
                                            else num_head]
                sh_emb = sh_emb.to(self._device)
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                    if (j + 1) * batch_size < num_tail \
                                                    else num_tail]
                    st_emb = st_emb.to(self._device)
                    s_score.append(sim_func(sh_emb, st_emb).to(th.device('cpu')))
                score.append(th.cat(s_score, dim=1))
            score = th.cat(score, dim=0)

            if bcast is False:
                result = []
                idx = th.arange(0, num_head * num_tail)
                score = th.reshape(score, (num_head * num_tail, ))

                topk_score, topk_sidx = th.topk(score,
                                                k=topk if score.shape[0] > topk else score.shape[0],
                                                dim=0)
                sidx = th.argsort(topk_score, dim=0, descending=True)
                score = topk_score[sidx]
                sidx = topk_sidx[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                result.append((head[head_idx],
                               tail[tail_idx],
                               score))

            else: # bcast at head
                result = []
                for i in range(num_head):
                    i_score = score[i]

                    topk_score, topk_sidx = th.topk(i_score,
                                                    k=topk if i_score.shape[0] > topk else i_score.shape[0],
                                                    dim=0)
                    sidx = th.argsort(topk_score, dim=0, descending=True)
                    i_score = topk_score[sidx]
                    idx = topk_sidx[sidx]

                    result.append((th.full((topk,), head[i], dtype=head[i].dtype),
                                  tail[idx],
                                  i_score))

        return result

    def embed_sim(self, left=None, right=None, embed_type='entity', sfunc='cosine', bcast=False, pair_ws=False, topk=10):
        """ Finds the most similar entity/relation embeddings for
        some pre-defined similarity functions given a set of
        entities or relations.

        Parameters
        ----------
        left: th.Tensor
            A tensor of left object id.

        right: th.Tensor
            A tensor of right object id.

        embed_type: str
            Whether it is using entity embedding or relation embedding.
            If `entity`, it is entity embedding.
            If 'relation', it is relation embedding.

        sfunc: str
            What kind of similarity function is used in ranking and will be output:

              * cosine: use cosine similarity, score = $\frac{x \cdot y}{||x||_2||y||_2}$'

              * l2: use l2 similarity, score = -$||x - y||_2$

              * l1: use l1 similarity, score = -$||x - y||_1$

              * dot: use dot product similarity, score = $x \cdot y$

              * ext_jaccard: use extended jaccard similarity, score = $\frac{x \cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \cdot y}$

        bcast: bool
            If True, both left and right objects are provided as L and R,, and we calculate topK for each element in L:

                * 'result = topK([score(l_i, r_j) for r_j in R]) for l_j in L, the result shape will be (sizeof(L), K)

            Default: False

        pair_ws: bool
            If True, both left and right objects are provided with the same length N, and we will calculate the similarity pair by pair:

              * result = topK([score(l_i, r_i)]) for i in N, the result shape will be (K,)

            Default: False

        topk: int
            Return top k results

        Note
        ----
        If both bcast and pair_ws is False, both left and right objects are provided as L and R,
        and we calculate all possible combinations of (l_i, r_j):
        ``result = topK([[score(l_i, rj) for l_i in L] for r_j in R])``,
        the result shape will be (K,)

        Return
        ------
        A list of (left_idx, right_idx, sim_score)
        """
        if embed_type == 'entity':
            emb = self.entity_embed
        elif embed_type == 'relation':
            emb = self.relation_embed
        else:
            assert False, 'emb should entity or relation'

        return self._embed_sim(head=left,
                               tail=right,
                               emb=emb,
                               sfunc=sfunc,
                               bcast=bcast,
                               pair_ws=pair_ws,
                               topk=topk)

    @property
    def model_name(self):
        return self._model_name

    @property
    def entity_embed(self):
        return self._entity_emb.emb

    @property
    def relation_embed(self):
        return self._relation_emb.emb

    @property
    def num_entity(self):
        return -1 if self.entity_embed is None else self.entity_embed.shape[0]

    @property
    def num_rel(self):
        return -1 if self.relation_embed is None else self.relation_embed.shape[0]

    @property
    def graph(self):
        return self._g

class KGEModel(BasicGEModel):
    """ Basic Knowledge Graph Embedding Model
    """
    def __init__(self, device, model_name, score_func):
        super(KGEModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        entity_emb_file = 'entity.npy'
        relation_emb_file = 'relation.npy'
        self._entity_emb.load(model_path, entity_emb_file)
        self._relation_emb.load(model_path, relation_emb_file)
        self._score_func.load(model_path, self.model_name)

class TransEModel(KGEModel):
    """ TransE Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE'
        score_func = TransEScore(gamma, 'l2')
        self._gamma = gamma
        super(TransEModel, self).__init__(device, model_name, score_func)

class TransE_l2Model(KGEModel):
    """ TransE_l2 Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE_l2'
        score_func = TransEScore(gamma, 'l2')
        self._gamma = gamma
        super(TransE_l2Model, self).__init__(device, model_name, score_func)

class TransE_l1Model(KGEModel):
    """ TransE_l1 Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE_l1'
        score_func = TransEScore(gamma, 'l1')
        self._gamma = gamma
        super(TransE_l1Model, self).__init__(device, model_name, score_func)

class TransRModel(KGEModel):
    """ TransR Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransR'
        # TransR score initialization is done at fit or load model
        projection_emb = KGEmbedding(device)
        score_func = TransRScore(gamma, projection_emb, -1, -1)
        self._gamma = gamma
        super(TransRModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        super(TransRModel, self).load(model_path)
        self._score_func.relation_dim = self._relation_emb.emb.shape[1]
        self._score_func.entity_dim = self._entity_emb.emb.shape[1]

class DistMultModel(KGEModel):
    """ DistMult Model
    """
    def __init__(self, device):
        model_name = 'DistMult'
        score_func = DistMultScore()
        super(DistMultModel, self).__init__(device, model_name, score_func)

class ComplExModel(KGEModel):
    """ ComplEx Model
    """
    def __init__(self, device):
        model_name = 'ComplEx'
        score_func = ComplExScore()
        super(ComplExModel, self).__init__(device, model_name, score_func)

class RESCALModel(KGEModel):
    """ RESCAL Model
    """
    def __init__(self, device):
        model_name = 'RESCAL'
        score_func = RESCALScore(-1, -1)
        super(RESCALModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        super(RESCALModel, self).load(model_path)
        self._score_func.entity_dim = self._entity_emb.emb.shape[1]
        self._score_func.relation_dim = self._relation_emb.emb.shape[1] // self._score_func.entity_dim

class RotatEModel(KGEModel):
    """ RotatE Model
    """
    def __init__(self, device, gamma):
        model_name = 'RotatE'
        self._gamma = gamma
        score_func = RotatEScore(gamma, 0)
        super(RotatEModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        super(RotatEModel, self).load(model_path)
        # retrive emb_init, which is used in scoring func
        entity_dim = self._entity_emb.emb.shape[1]
        hidden_dim = entity_dim // 2
        emb_init = (self._gamma + EMB_INIT_EPS) / hidden_dim
        self._score_func.emb_init = emb_init

class GNNModel(BasicGEModel):
    """ Basic GNN Model
    """
    def __init__(self, device, model_name, gamma=0):
        if model_name == 'TransE' or model_name == 'TransE_l2':
            score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            score_func = TransEScore(gamma, 'l1')
        elif model_name == 'DistMult':
            score_func = DistMultScore()
        else:
            assert model_name in ['TransE', 'TransE_l2', 'TransE_l1', 'DistMult'], \
                "For general purpose Scoring function for GNN, we only support TransE_l1, TransE_l2" \
                "DistMult, but {} is given.".format(model_name)

        super(GNNModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        entity_emb_file = 'entity.npy'
        relation_emb_file = 'relation.npy'
        self._entity_emb.load(model_path, entity_emb_file)
        self._relation_emb.load(model_path, relation_emb_file)


class GEModel(ABC, BasicGEModel):
    """ Graph Embedding model general framework
    User need to implement abstract method by their own
    """

    def __init__(self, args, device, model_name, score_func):
        self.args = args
        self.has_edge_importance = args.has_edge_importance
        self.lr = args.lr
        self.dist_train = (args.num_node * args.num_proc) != 1
        self.hidden_dim = args.hidden_dim
        self._optim = None
        self.regularizer = Regularizer(args.regularization_coef, args.regularization_norm)
        self._loss_gen = LossGenerator(args,
                                       args.loss_genre,
                                       args.neg_adversarial_sampling,
                                       args.adversarial_temperature,
                                       args.pairwise,
                                       args.label_smooth)

        # group embedding with learnable parameters to facilitate save, load, share_memory, etc
        self._entity_related_emb = dict()
        self._relation_related_emb = dict()
        self._torch_model = dict()
        self._global_relation_related_emb = None
        super(GEModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        """ Load all related parameters for embeddings.

        Parameters
        ----------
        model_path: str
            The path where all embeddings are stored
        """
        modules = [self._entity_related_emb, self._relation_related_emb, self._torch_model]
        for m in modules:
            for name, emb in m.items():
                emb_file = name + ('.pth' if m is self._torch_model else '.npy')
                emb.load(model_path, emb_file)

    def save(self, emap_file, rmap_file):
        """ Save embeddings related to entity, relation and score function.

        Parameters
        ----------
        emap_file
        rmap_file
        """
        args = self.args
        model_path = args.save_path

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        print('Save model to {}'.format(args.save_path))

        modules = [self._entity_related_emb,
                   self._relation_related_emb if self._global_relation_related_emb is None else self._global_relation_related_emb,
                   self._torch_model]
        for m in modules:
            for name, emb in m.items():
                emb_file = name + ('.pth' if m is self._torch_model else '.npy')
                emb.save(model_path, emb_file)

        # We need to save the model configurations as well.
        conf_file = os.path.join(args.save_path, 'config.json')
        dict = {}
        config = args
        dict.update(vars(config))
        dict.update({'emp_file': emap_file,
                     'rmap_file': rmap_file})
        with open(conf_file, 'w') as outfile:
            json.dump(dict, outfile, indent=4)

    def share_memory(self):
        """ dglke shares entity-related and relation-related embeddings across GPUs to accelerate training.
        """
        modules = [self._entity_related_emb, self._relation_related_emb]
        for m in modules:
            for emb in m.values():
                emb.share_memory()

    def train(self):
        """ Enable gradient backpropagation through computation graph.
        """
        modules = [self._entity_related_emb, self._relation_related_emb, self._torch_model]
        if self._global_relation_related_emb is not None:
            modules += [self._global_relation_related_emb]
        for m in modules:
            for emb in m.values():
                emb.train()

    def eval(self):
        """ Disable gradient backpropagation. When doing test, call this method to save GPU memory.
        """
        modules = [self._entity_related_emb, self._relation_related_emb, self._torch_model]
        if self._global_relation_related_emb is not None:
            modules += [self._global_relation_related_emb]
        for m in modules:
            for emb in m.values():
                emb.eval()

    def get_param_list(self):
        """ Get trainable parameters for weight regularization

        Returns
        -------
        a list of trainable parameters

        """
        param_list = []
        modules = [self._entity_related_emb, self._relation_related_emb]

        for m in modules:
            for emb in m.values():
                param_list += [emb.curr_emb()]

        for torch_module in self._torch_model.values():
            for params in torch_module.parameters():
                param_list += [params]

        return param_list

    def update(self, gpu_id):
        """ update all the parameters where there is gradient in their tensor in a sparse manner.

        Parameters
        ----------
        gpu_id: int
            Which gpu to accelerate the calculation. If -1 is provided, cpu is used.
        """
        modules = [self._entity_related_emb, self._relation_related_emb]
        for m in modules:
            for emb in m.values():
                emb.update(gpu_id)

        if self._optim is not None:
            self._optim.step()

    def create_async_update(self):
        for emb in self._entity_related_emb.values():
            emb.create_async_update()

    def finish_async_update(self):
        for emb in self._entity_related_emb.values():
            emb.finish_async_update()

    def test(self):
        # put it in train
        args = self.args
        if args.save_log:
            log_file = 'log.txt'
            result_file = 'result.txt'
            Logger.log_path = os.path.join(args.save_path, log_file)
            Logger.result_path = os.path.join(args.save_path, result_file)
        init_time_start = time.time()
        # load dataset and samplers
        dataset = get_dataset(args.data_path,
                              args.dataset,
                              args.format,
                              args.delimiter,
                              args.data_files,
                              args.has_edge_importance)

        if args.neg_sample_size_eval < 0:
            args.neg_sample_size_eval = dataset.n_entities
        args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
        # We need to ensure that the number of processes should match the number of GPUs.
        if len(args.gpu) > 1 and args.num_proc > 1:
            assert args.num_proc % len(args.gpu) == 0, \
                'The number of processes needs to be divisible by the number of GPUs'

        if args.neg_deg_sample_eval:
            assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

        # need to change to num_nodes if use 0.6.0 dgl version
        self.initialize(dataset.n_entities, dataset.n_relations, args.init_strat)
        self.categorize_embedding()
        if self.dist_train:
            # share memory for multiprocess to access
            self.share_memory()

        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc

        assert dataset.test is not None, 'test set is not provided'
        eval_dataset = EvalDataset(dataset, args)
        self.attach_graph(eval_dataset.g)
        self.load(args.save_path)
        print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

        # test
        start = time.time()
        self.eval()
        if args.num_test_proc > 1:
            queue = mp.Queue(args.num_test_proc)
            procs = []
            for i in range(args.num_test_proc):
                proc = mp.Process(target=self.eval_proc, args=(i, eval_dataset, 'test', queue))
                procs.append(proc)
                proc.start()

            metrics = {}
            logs = []
            for i in range(args.num_test_proc):
                log = queue.get()
                logs = logs + log

            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            print("-------------- Test result --------------")
            for k, v in metrics.items():
                print('Test average {} : {}'.format(k, v))
            print("-----------------------------------------")

            for proc in procs:
                proc.join()
        else:
            self.eval_proc(rank=0, eval_dataset=eval_dataset, mode='test')
        print('testing takes {:.3f} seconds'.format(time.time() - start))

    def fit(self):
        """ The whole process for model to be trained, validated, and tested
        """
        # put it in train
        args = self.args
        prepare_save_path(args)
        if args.save_log:
            log_file = 'log.txt'
            result_file = 'result.txt'
            Logger.log_path = os.path.join(args.save_path, log_file)
            Logger.result_path = os.path.join(args.save_path, result_file)
            print = Logger.print
        init_time_start = time.time()
        # load dataset and samplers
        dataset = get_dataset(args.data_path,
                              args.dataset,
                              args.format,
                              args.delimiter,
                              args.data_files,
                              args.has_edge_importance)

        if args.neg_sample_size_eval < 0:
            args.neg_sample_size_eval = dataset.n_entities
        args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
        args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
        # We should turn on mix CPU-GPU training for multi-GPU training.
        if len(args.gpu) > 1:
            args.mix_cpu_gpu = True
            if args.num_proc < len(args.gpu):
                args.num_proc = len(args.gpu)
        # We need to ensure that the number of processes should match the number of GPUs.
        if len(args.gpu) > 1 and args.num_proc > 1:
            assert args.num_proc % len(args.gpu) == 0, \
                'The number of processes needs to be divisible by the number of GPUs'

        # TODO: lingfei - degree based sampling is not supported currently
        if args.neg_deg_sample_eval:
            assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

        # partition training dataset here
        train_dataset = TrainDataset(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)
        args.strict_rel_part = args.mix_cpu_gpu and (train_dataset.cross_part is False)
        args.soft_rel_part = args.mix_cpu_gpu and args.rel_part and train_dataset.cross_part is True

        self.initialize(dataset.n_entities, dataset.n_relations, args.init_strat)
        self.categorize_embedding()

        eval_dataset = None
        if args.valid or args.test:
            if len(args.gpu) > 1:
                args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
            else:
                args.num_test_proc = args.num_proc
            if args.valid:
                assert dataset.valid is not None, 'validation set is not provided'
            if args.test:
                assert dataset.test is not None, 'test set is not provided'
            eval_dataset = EvalDataset(dataset, args)
            self.attach_graph(eval_dataset.g)

        if self.dist_train:
            # share memory for multiprocess to access
            self.share_memory()

        emap_file = dataset.emap_fname
        rmap_file = dataset.rmap_fname

        print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))
        # print configuration
        print('-' * 50)
        for k, v in vars(args).items():
            print('{} : {}'.format(k, v))
        print('-' * 50)
        # train
        start = time.time()
        rel_parts = train_dataset.rel_parts if args.strict_rel_part or args.soft_rel_part else None
        cross_rels = train_dataset.cross_rels if args.soft_rel_part else None

        self.train()

        if args.num_proc > 1:
            processes = []
            barrier = mp.Barrier(args.num_proc)
            for rank in range(args.num_proc):
                p = mp.Process(target=self.train_mp,
                               args=(rank, train_dataset, eval_dataset, rel_parts, cross_rels, barrier))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            self.train_proc(0, train_dataset, eval_dataset, rel_parts, cross_rels)
        print('training takes {} seconds'.format(time.time() - start))

        if not args.no_save_emb:
            self.save(emap_file, rmap_file)

        # test
        self.eval()
        if args.test:
            start = time.time()
            if args.num_test_proc > 1:
                queue = mp.Queue(args.num_test_proc)
                procs = []
                for i in range(args.num_test_proc):
                    proc = mp.Process(target=self.eval_mp, args=(i, eval_dataset, 'test', queue))
                    procs.append(proc)
                    proc.start()

                metrics = {}
                logs = []
                for i in range(args.num_test_proc):
                    log = queue.get()
                    logs = logs + log

                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                print("-------------- Test result --------------")
                for k, v in metrics.items():
                    print('Test average {} : {}'.format(k, v))
                print("-----------------------------------------")

                for proc in procs:
                    proc.join()
            else:
                self.eval_proc(rank=0, eval_dataset=eval_dataset, mode='test')
            print('testing takes {:.3f} seconds'.format(time.time() - start))

    def compute_ranking(self, pos_score, neg_score, data, mode='tail', eval_filter=True, self_loop_filter=True):
        head, rel, tail, neg = data['head'], data['rel'], data['tail'], data['neg']
        b_size = data['head'].shape[0]
        pos_score = pos_score.view(b_size, -1)
        neg_score = neg_score.view(b_size, -1)
        ranking = th.zeros(b_size, 1, device=th.device('cpu'))
        log = []
        for i in range(b_size):
            cand_idx = (neg_score[i] >= pos_score[i]).nonzero(as_tuple=False).cpu()
            # there might be precision error where pos_score[i] actually equals neg_score[i, pos_entity[i]]
            # we explicitly add this index to cand_idx to overcome this issue
            if mode == 'tail':
                if tail[i] not in cand_idx:
                    cand_idx = th.cat([cand_idx, tail[i].detach().cpu().view(-1, 1)], dim=0)
                # here we filter out self-loop(head-relation-head)
                if self_loop_filter and head[i] in cand_idx:
                    cand_idx = cand_idx[cand_idx != head[i]].view(-1, 1)
            else:
                if head[i] not in cand_idx:
                    cand_idx = th.cat([cand_idx, head[i].detach().cpu().view(-1, 1)], dim=0)
                if self_loop_filter and tail[i] in cand_idx:
                    cand_idx = cand_idx[cand_idx != tail[i]].view(-1, 1)
            cand_num = len(cand_idx)
            if not eval_filter:
                ranking[i] = cand_num
                continue
            if mode is 'tail':
                select = self.graph.has_edges_between(head[i], neg[cand_idx[:, 0]]).nonzero(as_tuple=False)[:, 0]
                if len(select) > 0:
                    select_idx = cand_idx[select].view(-1)
                    uid, vid, eid = self.graph.edge_ids(head[i], select_idx, return_uv=True)
                else:
                    raise ValueError('at least one element should have the same score with pos_score. That is itself!')
            else:
                select = self.graph.has_edges_between(neg[cand_idx[:, 0]], tail[i]).nonzero(as_tuple=False)[:, 0]
                if len(select) > 0:
                    select_idx = cand_idx[select].view(-1)
                    uid, vid, eid = self.graph.edge_ids(select_idx, tail[i], return_uv=True)
                else:
                    raise ValueError('at least one element should have the same score with pos_score. That is itself!')

            rid = self.graph.edata[self._etid_field][eid]
            #  - 1 to exclude rank for positive score itself
            cand_num -= th.sum(rid == rel[i]) - 1
            ranking[i] = cand_num

        for i in range(b_size):
            ranking_i = get_scalar(ranking[i])
            log.append({
                'MRR': 1.0 / ranking_i,
                'MR': float(ranking_i),
                'HITS@1': 1.0 if ranking_i <= 1 else 0.0,
                'HITS@3': 1.0 if ranking_i <= 3 else 0.0,
                'HITS@10': 1.0 if ranking_i <= 10 else 0.0
            })

        return ranking, log

    # misc for DataParallelTraining
    def setup_model(self, rank, world_size, gpu_id):
        """ Set up score function for DistributedDataParallel.
        As score function is a dense model, if we need to parallelly train/eval the mode, we need to put the model into different gpu devices

        Parameters
        ----------
        rank : int
            process id in regards of world size
        world_size : int
            total number of process
        gpu_id : int
            which device should the model be put to, -1 for cpu otherwise gpu

        """
        args = self.args

        for name, module in self._torch_model.items():
            self._torch_model[name] = to_device(module, gpu_id)

        if self.dist_train and len(self._torch_model.items()) != 0:
            # configure MASTER_ADDR and MASTER_PORT manually in command line
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '8888'
            dist.init_process_group('nccl', rank=rank, world_size=world_size)

        # make DDP model
        # broadcast_buffers=False to enable batch normalization
        params = []
        for name, module in self._torch_model.items():
            self._torch_model[name] = module if not self.dist_train else DistributedDataParallel(module,
                                                                                                 device_ids=[gpu_id],
                                                                                                 broadcast_buffers=False)
            for param in self._torch_model[name].parameters():
                params += [param]

        if len(params) != 0:
            if args.optimizer == 'Adagrad':
                self._optim = Adagrad(params, lr=self.lr)
            elif args.optimizer == 'Adam':
                self._optim = Adam(params, lr=self.lr)

    def cleanup(self):
        """ destroy parallel process if necessary

        Parameters
        ----------
        dist_train : bool
            whether it's distributed training or not

        """
        if self.dist_train and len(self._torch_model) != 0:
            dist.destroy_process_group()
        else:
            pass

    def prepare_relation(self, device=None):
        local_emb = {}
        for k, v in self._relation_related_emb.items():
            local_emb[k] = v.clone(device=device)
        self._global_relation_related_emb = self._relation_related_emb
        self._relation_related_emb = local_emb

    def prepare_cross_rels(self, cross_rels):
        for k, v in self._relation_related_emb.items():
            v.setup_cross_rels(cross_rels, self._global_relation_related_emb[k])

    def writeback_relation(self, rank=0, rel_parts=None):
        args = self.args
        idx = rel_parts[rank]
        for name, embeddings in self._relation_related_emb.items():
            if args.soft_rel_part:
                local_idx = embeddings.get_noncross_idx(idx)
            else:
                local_idx = idx
            #  MARK - TBD, whether detach here
            self._global_relation_related_emb[name].emb[local_idx] = embeddings.emb.detach().clone().cpu()[local_idx]

    def train_proc(self, rank, train_dataset, eval_dataset, rel_parts=None, cross_rels=None, barrier=None):
        """ training process for fit(). it will read data, forward embedding data, compute loss and update param using gradients

        Parameters
        ----------
        rank : int
            process id in regards of world size
        train_dataset : KGDataset
            dataset used for training
        eval_dataset : KGDataset
            dataset used for evaluation
        """
        # setup
        if rank == 0:
            profiler = Profiler()
            profiler.start()
        args = self.args
        world_size = args.num_proc * args.num_node
        if len(args.gpu) > 0:
            gpu_id = args.gpu[rank % len(args.gpu)] if args.num_proc > 1 else args.gpu[0]
        else:
            gpu_id = -1

        # setup optimizer, load embeddings into gpu, enable async_update
        if args.async_update:
            self.create_async_update()
        if args.strict_rel_part or args.soft_rel_part:
            self.prepare_relation(th.device(f'cuda:{gpu_id}'))
        if args.soft_rel_part:
            self.prepare_cross_rels(cross_rels)

        self.setup_model(rank, world_size, gpu_id)

        logs = []
        for arg in vars(args):
            logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

        train_start = start = time.time()
        sample_time = 0
        update_time = 0
        forward_time = 0
        backward_time = 0
        batch_size = args.batch_size // args.neg_sample_size

        partition_dataset = PartitionChunkDataset(train_dataset, rank, world_size, 'train', args.neg_sample_size, args.max_step, batch_size)
        partition_dataset.pin_memory()
        sampler = SequentialTotalSampler(batch_size=batch_size, max_step=args.max_step)
        dataloader = DataLoader(dataset=partition_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler,
                                num_workers=0,
                                drop_last=False,
                                pin_memory=True)
        data_iter = iter(dataloader)
        step_range = trange(0, args.max_step, desc='train') if (rank == 0 and args.tqdm) else range(0, args.max_step)
        for step in step_range:
            neg_type = 'head' if step % 2 == 0 else 'tail'
            start1 = time.time()
            # get pos training data
            data = next(data_iter)
            data = {k: v.view(-1) for k, v in data.items()}
            sample_time += time.time() - start1
            loss, log = self.train_forward(data, neg_type, gpu_id)
            if rank == 0 and args.tqdm:
                step_range.set_postfix(loss=f'{loss.item():.4f}')
            forward_time += time.time() - start1

            start1 = time.time()
            if self._optim is not None:
                self._optim.zero_grad()
            loss.backward()

            backward_time += time.time() - start1
            # update embedding & dense_model using different optimizer, for dense_model, use regular pytorch optimizer
            # for embedding, use built-in Adagrad sync/async optimizer
            start1 = time.time()
            self.update(gpu_id)
            update_time += time.time() - start1
            logs.append(log)

            if args.force_sync_interval > 0 and (step + 1) % args.force_sync_interval == 0:
                barrier.wait()

            if (step + 1) % args.log_interval == 0:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                             time.time() - start))
                print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

            if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and eval_dataset is not None:
                valid_start = time.time()
                # for async update
                if args.strict_rel_part or args.soft_rel_part:
                    self.writeback_relation(rank, rel_parts)
                # forced sync for validation
                if self.dist_train:
                    barrier.wait()
                self.eval_proc(rank, eval_dataset, mode='valid')
                self.train()
                print('[proc {}]validation take {:.3f} seconds.'.format(rank, time.time() - valid_start))
                if args.soft_rel_part:
                    self.prepare_cross_rels(cross_rels)
                if self.dist_train:
                    barrier.wait()

        print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
        if args.async_update:
            self.finish_async_update()
        if args.strict_rel_part or args.soft_rel_part:
            self.writeback_relation(rank, rel_parts)
        self.cleanup()
        if rank == 0:
            profiler.stop()
            print(profiler.output_text(unicode=False, color=False))

    def eval_proc(self, rank, eval_dataset, mode='valid', queue=None):
        if rank == 0:
            profiler = Profiler()
            profiler.start()
        args = self.args
        if len(args.gpu) > 0:
            gpu_id = args.gpu[rank % len(args.gpu)] if args.num_proc > 1 else args.gpu[0]
        else:
            gpu_id = -1

        world_size = args.num_proc * args.num_node

        if mode is not 'valid':
            self.setup_model(rank, world_size, gpu_id)
            self.eval()

        partition_dataset = PartitionChunkDataset(eval_dataset, rank, world_size, mode, None, None, None)
        partition_dataset.pin_memory()
        pos_dataloader = DataLoader(dataset=partition_dataset,
                                    batch_size=args.batch_size_eval,
                                    shuffle=False,
                                    num_workers=0,
                                    drop_last=False,
                                    pin_memory=True)
        data = dict()
        data['neg'] = eval_dataset.g.nodes().clone()
        with th.no_grad():
            logs = []
            iterator = tqdm(iter(pos_dataloader), desc='evaluation') if (rank == 0 and args.tqdm) else iter(pos_dataloader)
            for pos_data in iterator:
                # update data[-1] to all the nodes in the graph to perform corruption for all the nodes
                data.update(pos_data)
                log = self.test_forward(data, gpu_id)
                logs += log

            metrics = {}
            if len(logs) > 0:
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            if queue is not None:
                queue.put(logs)
            else:
                for k, v in metrics.items():
                    print('[{}]{} average {}: {}'.format(rank, mode, k, v))
                Logger.save_result(metrics)

        if mode is not 'valid':
            self.cleanup()
        if rank == 0:
            profiler.stop()
            print(profiler.output_text(unicode=False, color=False))

    def train_forward(self, data, neg_type, gpu_id):
        args = self.args
        chunk_size = args.neg_sample_size
        neg_sample_size = args.neg_sample_size
        pos_emb = self.acquire_embedding(data=data, gpu_id=gpu_id, pos=True, train=True, neg_type=neg_type)

        edge_impts = to_device(data['impts'], gpu_id) if args.has_edge_importance else None

        pos_score = self.pos_forward(pos_emb)

        neg_emb = self.acquire_embedding(data=data, gpu_id=gpu_id, pos=False, train=True, neg_type=neg_type)

        neg_score = self.neg_forward(pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=True)
        neg_score = neg_score.reshape(-1, neg_sample_size)

        loss, log = self._loss_gen.get_total_loss(pos_score, neg_score, edge_impts)
        reg, reg_log = self.regularizer(self.get_param_list())

        loss += reg
        log.update(reg_log)

        return loss, log

    def test_forward(self, data, gpu_id):
        args = self.args
        log = []
        pos_emb = self.acquire_embedding(data, gpu_id, pos=True, train=False)
        neg_emb = self.acquire_embedding(data, gpu_id, pos=False, train=False)
        batch_size = data['head'].shape[0]
        num_node = len(data['neg'])
        pos_score = self.pos_forward(pos_emb)
        neg_score_corr_head = self.neg_forward(pos_emb, neg_emb, 'head', chunk_size=batch_size, neg_sample_size=num_node, train=False)
        neg_score_corr_tail = self.neg_forward(pos_emb, neg_emb, 'tail', chunk_size=batch_size, neg_sample_size=num_node, train=False)
        ranking_corr_tail, log_corr_tail = self.compute_ranking(pos_score, neg_score_corr_tail, data, mode='tail',
                                                                eval_filter=args.eval_filter,
                                                                self_loop_filter=args.self_loop_filter)
        ranking_corr_head, log_corr_head = self.compute_ranking(pos_score, neg_score_corr_head, data, mode='head',
                                                                eval_filter=args.eval_filter,
                                                                self_loop_filter=args.self_loop_filter)
        log += log_corr_tail
        log += log_corr_head
        return log

    def prepare_data(self, pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=True):
        """ Prepare positive/negative embedding data for training/evaluation. This is the place to reshape tensor
        to enable operation like bmm takes place.

        Parameters
        ----------
        pos_emb: dict
            dictionary containing all positive embeddings involved.
        neg_emb: dict
            dictionary containing all negative embeddings involved.
        neg_type: str
            choice: ['head', 'tail'], for this batch, triples are corrupted by neg_type.
        chunk_size: int
            normally to reshape positive embeddings from [batch, embbeding_size]
            to [\frac{batch}{chunk_size}, chunk_size, embedding_size]
        neg_sample_size: int
            normally to reshape negative embeddings from [batch, embedding_size]
            to [\frac{batch}{neg_sample_size}, neg_sample_size, embedding_size]
        train: bool
            prepare data for training or evaluation. Model for evaluation might have different behavior.

        Returns
        -------
        th.Tensor: reshaped pos_emb
        th.Tensor: reshaped neg_emb

        """
        return pos_emb, neg_emb

    @thread_wrapped_func
    def train_mp(self, rank, train_dataset, eval_dataset, rel_parts=None, cross_rels=None, barrier=None):
        args = self.args
        if args.num_proc > 1:
            th.set_num_threads(args.num_thread)
        self.train_proc(rank, train_dataset, eval_dataset, rel_parts, cross_rels, barrier)

    @thread_wrapped_func
    def eval_mp(self, rank, eval_dataset, mode='valid', queue=None):
        args = self.args
        if args.num_proc > 1:
            th.set_num_threads(args.num_thread)
        self.eval_proc(rank, eval_dataset, mode, queue)

    @abstractmethod
    def categorize_embedding(self):
        pass

    @abstractmethod
    def initialize(self, n_entities, n_relations, init_strat='uniform'):
        pass

    @abstractmethod
    def pos_forward(self, pos_emb):
        pass

    @abstractmethod
    def neg_forward(self, pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=True):
        pass

    @abstractmethod
    def acquire_embedding(self, data, gpu_id=-1, pos=True, train=True, neg_type='head'):
        pass

