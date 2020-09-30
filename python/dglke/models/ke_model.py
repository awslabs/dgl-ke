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

class BasicGEModel(object):
    """ Basic Graph Embeding Model
    """
    def __init__(self, device):
        self._g = None

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
        else:
            self._dataset = data
            self._g = self.load_dataset()

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
        g = dgl.graph((src, dst)
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
        pass

    def fit(self):
        """ Start training
        """
        assert False, 'Not support training now'

    def eval(self):
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

    def _embed_sim(self, head, tail, emb, sfunc='cosine', bcast=False, pair_ws=False, k=10):

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
            Whether it is using entity embedding or relation embedding

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

    @property
    def entity_embed(self):
        pass

    @property
    def relation_embed(self):
        pass