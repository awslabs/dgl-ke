# -*- coding: utf-8 -*-
#
# train.py
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

import os
import time
import argparse
import numpy as np

import dgl.backend as F

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    from .mxnet.tensor_models import logsigmoid
    from .mxnet.tensor_models import norm_l1
else:
    from .pytorch.tensor_models import logsigmoid
    from .pytorch.tensor_models import norm_l1
from .models import InferModel

class ScoreInfer(object):
    """ Calculate score of triplet (h, r, t) based on pretained KG embeddings
        using specified score_function

    Parameters
    ---------
    config : dict
        Containing KG model information

    model_path : str
        path storing the model (pretrained embeddings)

    score_func : str
        What kind of score is used,
            l1: score = $|x|$
            logsigmoid: score $log(sigmoid(x))
    """
    def __init__(self, device, config, model_path, sfunc='L1'):
        assert score_func in ['l1', 'logsigmoid'], 'score function should be l1 or logsigmoid'

        self.device = 'cpu' if device < 0 else device
        self.load_model(config, model_path)
        self.sfunc = sfunc
        if sfunc == 'l1'
            self.score_func = norm_l1
        else:
            self.score_func = logsigmoid

    def load_model(self, config, model_path):
        model = InferModel(device=self.device,
                           model_name=config['model'],
                           hidden_dim=config['emd_size'],
                           double_entity_emb=config['double_ent'],
                           double_relation_emb=config['double_rel'],
                           gamma=config['gamma'])
        dataset = config['dataset']
        model.load_emb(model_path, dataset)
        self.model = model

    def topK(self, head=None, rel=None, tail=None, bcast=None, k=10):
        if head is None:
            head = F.arange(0, self.model.num_entity)
        else:
            head = F.zerocopy_from_numpy(head)
        if rel is None:
            rel = F.arange(0, self.model.num_rel)
        else:
            rel = F.zerocopy_from_numpy(rel)
        if tail is None:
            tail = F.arange(0, self.model.num_entity)
        else:
            tail = F.zerocopy_from_numpy(tail)

        num_head = F.shape(head)[0]
        num_rel = F.shape(rel)[0]
        num_tail = F.shape(tail)[0]

        if bcast == 'none':
            result = []
            raw_score = self.model.score(head, rel, tail)
            score = self.score_func(raw_score)
            idx = F.arange(0, num_head * num_rel * num_tail)
            
            if sfunc == 'l1':
                sidx = argsort(score, dim=1, descending=False)
            else:
                sidx = argsort(score, dim=1, descending=True)

            sidx = sidx[:k]
            score = score[sidx]
            idx = idx[sidx]
            tail_idx = idx % num_tail
            idx = idx / num_tail
            rel_idx = idx % num_rel
            idx = idx / num_rel
            head_idx = idx % num_head
            
            result.append((F.asnumpy(head[head_idx]),
                           F.asnumpy(rel[rel_idx]),
                           F.asnumpy(tail[tail_idx]),
                           F.asnumpy(score)))
        elif bcast == 'head':
            result = []
            for i in range(num_head):
                raw_score = self.model.score(head[i], rel, tail)
                score = self.score_func(raw_score)
                idx = F.arange(0, num_rel * num_tail)

                if sfunc == 'l1':
                    sidx = argsort(score, dim=1, descending=False)
                else:
                    sidx = argsort(score, dim=1, descending=True)

                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = idx / num_tail
                rel_idx = idx % num_rel

                result.append((np.full((k,), head[i]),
                               F.asnumpy(rel[rel_idx]),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif bcast == 'rel':
            result = []
            for i in range(num_rel):
                raw_score = self.model.score(head, rel[i], tail)
                score = self.score_func(raw_score)

                if sfunc == 'l1':
                    sidx = argsort(score, dim=1, descending=False)
                else:
                    sidx = argsort(score, dim=1, descending=True)

                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = idx / num_tail
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                               np.full((k,), rel[i]),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif bcast == 'tail':
            result = []
            for i in range(num_tail):
                raw_score = self.model.score(head, rel, tail[i])
                score = self.score_func(raw_score)

                if sfunc == 'l1':
                    sidx = argsort(score, dim=1, descending=False)
                else:
                    sidx = argsort(score, dim=1, descending=True)

                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                rel_idx = idx % num_rel
                idx = idx / num_rel
                head_idx = idx % num_head
                result.append((F.asnumpy(head[head_idx]),
                               F.asnumpy(rel[rel_idx]),
                               np.full((k,), tail[i]),
                               F.asnumpy(score)))
        else:
            assert False, 'unknow broadcast type {}'.format(bcast)

        return result


class EntitySimInfer():
    def __init__(self):
        pass
