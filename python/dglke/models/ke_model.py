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
"""
import os
from abc import abstractmethod, ABCMeta
import numpy as np
import dgl
import torch as th

from .pytorch.tensor_models import logsigmoid
from .pytorch.tensor_models import get_device
from .pytorch.tensor_models import norm
from .pytorch.tensor_models import get_scalar
from .pytorch.tensor_models import reshape
from .pytorch.tensor_models import cuda
from .pytorch.tensor_models import ExternalEmbedding
from .pytorch.tensor_models import InferEmbedding
from .pytorch.score_fun import *
from .pytorch.ke_tensor import KGEmbedding

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

    def attach_data(self, data, etid_field='tid', ntid_filed='ntid'):
        """ Attach dataset into Graph Embedding Model

        Parameter
        ----------
        data: KGDataset or DGLGraph
            Input data for knowledge graph
        etid_field: str
            Edge feature name storing the edge type id
        ntid_filed: str
            Node feature name storing the node type id

        Note
        ----
        If the input graph is DGLGraph, we assume that it uses a homogeneous graph
        to represent the heterogeneous graph. The edge type id is stored in etid_field
        and the node type id is stored in ntid_filed.
        """
        self._etid_field = etid_field
        self._ntid_filed = ntid_filed
        if isinstance(data, dgl.DGLGraph):
            self._g = data
            self._dataset_name = None
        else:
            self._dataset = data
            self._g = self.load_dataset()
            self._dataset_name = self._dataset.name

    def load_dataset(self):
        train = self._dataset.train
        valid = self._dataset.valid
        test = self._dataset.test
        src = [train[0]]
        etype_id = [train[1]]
        dst = [train[2]]
        self.num_train = len(train[0])
        if valid is not None:
            src.append(valid[0])
            etype_id.append(valid[1])
            dst.append(valid[2])
            self.num_valid = len(valid[0])
        else:
            self.num_valid = 0
        if test is not None:
            src.append(test[0])
            etype_id.append(test[1])
            dst.append(test[2])
            self.num_test = len(test[0])
        else:
            self.num_test = 0
        assert len(src) > 1, "we need to have at least validation set or test set."
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)
        g = dgl.graph((src, dst))
        g.edata[self._etid_field] = th.Tensor(etype_id)
        return g

    def load(self, model_path, entity_emb_file=None, relation_emb_file=None):
        """ Load Graph Embedding Model from model_path.

        All model related data is stored under model_path. If entity_emb_file and
        relation_emb_file are provided, entity embeddings and relation embeddings
        are directly loaded from them. Otherwise default file name is used (i.e.,
        $dataset$_$model_name$_entity.npy and $dataset$_$model_name$_relation.npy')

        Parameter
        ---------
        model_path : str
            Path to store the model information
        entity_emb_file : str
            Dedicated file to store entity embeddings. If None, use default name.
            Default: None.
        relation_emb_file : str
            Dedicated file to store relation embeddings. If None, use default name.
            Default: None.
        """
        pass

    def save(self, model_path, entity_emb_file=None, relation_emb_file=None):
        """ Save Graph Embedding Model into model_path.

        All model related data are saved under model_path. If entity_emb_file and
        relation_emb_file are provided, entity embeddings and relation embeddings
        are saved with the specified file names. Otherwise default file name is used
        (i.e., $dataset$_$model_name$_entity.npy and $dataset$_$model_name$_relation.npy')

        Parameter
        ---------
        model_path : str
            Path to store the model information
        entity_emb_file : str
            Dedicated file to store entity embeddings. If None, use default name.
            Default: None.
        relation_emb_file : str
            Dedicated file to store relation embeddings. If None, use default name.
            Default: None.
        """
        assert False, 'Not support training now'

    def fit(self, params):
        """ Start training
        """
        assert False, 'Not support training now'

    def eval(self, params):
        """ Start evaluation
        """
        assert False, 'Not support evaluation now'

    def link_predict(self, head=None, rel=None, tail=None, exec_mode='all', sfunc='none', topk=10, exclude_mode=None):
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
            head = F.arange(0, self.num_entity)
        else:
            head = F.tensor(head)
        if rel is None:
            rel = F.arange(0, self.num_rel)
        else:
            rel = F.tensor(rel)
        if tail is None:
            tail = F.arange(0, self.num_entity)
        else:
            tail = F.tensor(tail)

        num_head = F.shape(head)[0]
        num_rel = F.shape(rel)[0]
        num_tail = F.shape(tail)[0]

        if sfunc == 'none':
            sfunc = none
        else:
            sfunc = logsigmoid

        # if exclude_mode is not None, we need a graph to do the edge filtering
        assert (self.g is not None) or (exclude_mode is None),
            'If exclude_mode is not None, please use load_graph() to initialize ' \
            'a graph for edge filtering.'
        if exec_mode == 'triplet_wise':
            result = []
            assert num_head == num_rel, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'
            assert num_head == num_tail, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'

            raw_score = self._score_func(head, rel, tail, triplet_wise=True)
            score = sfunc(raw_score)
            idx = F.arange(0, num_head)

            sidx = F.argsort(score, dim=0, descending=True)
            sidx = sidx[:k]
            score = score[sidx]
            idx = idx[sidx]

            result.append((F.asnumpy(head[idx]),
                           F.asnumpy(rel[idx]),
                           F.asnumpy(tail[idx]),
                           F.asnumpy(score)))

        elif exec_mode == 'all':
            result = []
            raw_score = self._score_func(head, rel, tail)
            score = sfunc(raw_score)
            idx = F.arange(0, num_head * num_rel * num_tail)

            sidx = F.argsort(score, dim=0, descending=True)
            sidx = sidx[:k]
            score = score[sidx]
            idx = idx[sidx]

            tail_idx = idx % num_tail
            idx = floor_divide(idx, num_tail)
            rel_idx = idx % num_rel
            idx = floor_divide(idx, num_rel)
            head_idx = idx % num_head

            result.append((F.asnumpy(head[head_idx]),
                           F.asnumpy(rel[rel_idx]),
                           F.asnumpy(tail[tail_idx]),
                           F.asnumpy(score)))
        elif exec_mode == 'batch_head':
            result = []
            for i in range(num_head):
                raw_score = self._score_func(F.unsqueeze(head[i], 0), rel, tail)
                score = sfunc(raw_score)
                idx = F.arange(0, num_rel * num_tail)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                rel_idx = idx % num_rel

                result.append((np.full((k,), F.asnumpy(head[i])),
                               F.asnumpy(rel[rel_idx]),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif exec_mode == 'batch_rel':
            result = []
            for i in range(num_rel):
                raw_score = self._score_func(head, F.unsqueeze(rel[i], 0), tail)
                score = sfunc(raw_score)
                idx = F.arange(0, num_head * num_tail)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                               np.full((k,), F.asnumpy(rel[i])),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif exec_mode == 'batch_tail':
            result = []
            for i in range(num_tail):
                raw_score = self._score_func(head, rel, F.unsqueeze(tail[i], 0))
                score = sfunc(raw_score)
                idx = F.arange(0, num_head * num_rel)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                rel_idx = idx % num_rel
                idx = floor_divide(idx, num_rel)
                head_idx = idx % num_head
                result.append((F.asnumpy(head[head_idx]),
                               F.asnumpy(rel[rel_idx]),
                               np.full((k,), F.asnumpy(tail[i])),
                               F.asnumpy(score)))
        else:
            assert False, 'unknow execution mode type {}'.format(exec_mode)

        return result

    def _embed_sim(self, head, tail, emb, sfunc='cosine', bcast=False, pair_ws=False, k=10):
        batch_size=DEFAULT_INFER_BATCHSIZE
        if head is None:
            head = F.arange(0, emb.shape[0])
        else:
            head = F.tensor(head)
        if tail is None:
            tail = F.arange(0, emb.shape[0])
        else:
            tail = F.tensor(tail)

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
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sh_emb = F.copy_to(sh_emb, self._device)
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                st_emb = F.copy_to(st_emb, self._device)
                score.append(F.copy_to(sim_func(sh_emb, st_emb, pw=True), F.cpu()))
            score = F.cat(score, dim=0)

            sidx = F.argsort(score, dim=0, descending=True)
            sidx = sidx[:k]
            score = score[sidx]
            result.append((F.asnumpy(head[sidx]),
                           F.asnumpy(tail[sidx]),
                           F.asnumpy(score)))
        else:
            num_head = head.shape[0]
            num_tail = tail.shape[0]

            # chunked cal score
            score = []
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                            if (i + 1) * batch_size < num_head \
                                            else num_head]
                sh_emb = F.copy_to(sh_emb, self._device)
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                    if (j + 1) * batch_size < num_tail \
                                                    else num_tail]
                    st_emb = F.copy_to(st_emb, self._device)
                    s_score.append(F.copy_to(sim_func(sh_emb, st_emb), F.cpu()))
                score.append(F.cat(s_score, dim=1))
            score = F.cat(score, dim=0)

            if bcast is False:
                result = []
                idx = F.arange(0, num_head * num_tail)
                score = F.reshape(score, (num_head * num_tail, ))

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                sidx = sidx
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                           F.asnumpy(tail[tail_idx]),
                           F.asnumpy(score)))

            else: # bcast at head
                result = []
                for i in range(num_head):
                    i_score = score[i]

                    sidx = F.argsort(i_score, dim=0, descending=True)
                    idx = F.arange(0, num_tail)
                    i_idx = sidx[:k]
                    i_score = i_score[i_idx]
                    idx = idx[i_idx]

                    result.append((np.full((k,), F.asnumpy(head[i])),
                                  F.asnumpy(tail[idx]),
                                  F.asnumpy(i_score)))

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

        return self._embed_sim(head=head,
                               tail=tail,
                               emb=emb,
                               sfunc=sfunc,
                               bcast=bcast,
                               pair_ws=pair_ws,
                               k=k)

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

class KGEModel(BasicGEModel):
    """ Basic Knowledge Graph Embedding Model
    """
    def __init__(self, device, model_name, score_func):
        super(KGEModel, self).__init__(device, model_name, score_func)

    def load(self, model_path, entity_emb_file=None, relation_emb_file=None):
        entity_emb_file = entity_emb_file if entity_emb_file is not None \
                                          else self._dataset_name+'_'+self.model_name+'_entity.npy'
        relation_emb_file = relation_emb_file if relation_emb_file is not None \
                                              else self._dataset_name+'_'+self.model_name+'_relation.npy'
        self._entity_emb.load(model_path, entity_emb_file)
        self._relation_emb.load(model_path, relation_emb_file)
        self._score_func.load(model_path, dataset+'_'+self.model_name)

class TransEModel(BasicGEModel):
    """ TransE Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE'
        score_func = TransEScore(gamma, 'l2')
        self._gamma = gamma
        super(TransEModel, self).__init__(model_name, score_func)

class TransE_l2Model(BasicGEModel):
    """ TransE_l2 Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE_l2'
        score_func = TransEScore(gamma, 'l2')
        self._gamma = gamma
        super(TransE_l2Model, self).__init__(model_name, score_func)

class TransE_l1Model(BasicGEModel):
    """ TransE_l1 Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE_l1'
        score_func = TransEScore(gamma, 'l1')
        self._gamma = gamma
        super(TransE_l1Model, self).__init__(model_name, score_func)

class TransRModel(BasicGEModel):
    """ TransR Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransR'
        # TransR score initialization is done at fit or load model
        projection_emb = KGEmbedding(device)
        score_func = TransRScore(gamma, projection_emb, -1, -1)
        self._gamma = gamma
        super(TransRModel, self).__init__(model_name, _score_func)

    def load(self, model_path, entity_emb_file=None, relation_emb_file=None):
        super(TransRModel, self).load(model_path, entity_emb_file, relation_emb_file)
        self._score_func.relation_dim = self._relation_emb.emb.shape[1]
        self._score_func.entity_dim = self._entity_emb.emb.shape[1]

class DistMultModel(BasicGEModel):
    """ DistMult Model
    """
    def __init__(self, device):
        model_name = 'DistMult'
        score_func = DistMultScore()
        super(DistMultModel, self).__init__(model_name, score_func)

class ComplExModel(BasicGEModel):
    """ ComplEx Model
    """
    def __init__(self, device):
        model_name = 'ComplEx'
        score_func = ComplExScore()
        super(ComplExModel, self).__init__(model_name, score_func)

class RESCALModel(BasicGEModel):
    """ RESCAL Model
    """
    def __init__(self, device):
        model_name = 'RESCAL'
        score_func = RESCALScore(-1, -1)
        super(RESCALModel, self).__init__(model_name, score_func)

    def load(self, model_path, entity_emb_file=None, relation_emb_file=None):
        super(TransRModel, self).load(model_path, entity_emb_file, relation_emb_file)
        self._score_func.relation_dim = self._relation_emb.emb.shape[1]
        self._score_func.entity_dim = self._entity_emb.emb.shape[1]

class RotatEModel(BasicGEModel):
    """ RESCAL Model
    """
    def __init__(self, device, gamma):
        model_name = 'RotatE'
        score_func = RotatEcore(gamma, 0)
        self._gamma = gamma
        super(RotatEModel, self).__init__(model_name, score_func)

class GNNModel(BasicGEModel):
    """ Basic GNN Model
    """
    def __init__(self, device, model_name, gamma):
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
