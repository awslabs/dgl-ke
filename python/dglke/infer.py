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
from models import InferModel
if backend.lower() == 'mxnet':
    from models.mxnet.tensor_models import logsigmoid
    from models.mxnet.tensor_models import none
    from models.mxnet.tensor_models import cosine_dist
    from models.mxnet.tensor_models import l2_dist
    from models.mxnet.tensor_models import l1_dist
    from models.mxnet.tensor_models import dot_dist
    from models.mxnet.tensor_models import extended_jaccard_dist
else:
    from models.pytorch.tensor_models import logsigmoid
    from models.pytorch.tensor_models import none
    from models.pytorch.tensor_models import cosine_dist
    from models.pytorch.tensor_models import l2_dist
    from models.pytorch.tensor_models import l1_dist
    from models.pytorch.tensor_models import dot_dist
    from models.pytorch.tensor_models import extended_jaccard_dist

class ScoreInfer(object):
    """ Calculate score of triplet (h, r, t) based on pretained KG embeddings
        using specified score_function

    Parameters
    ---------
    device : int 
        Device to run the inference, -1 for CPU

    config : dict
        Containing KG model information

    model_path : str
        path storing the model (pretrained embeddings)

    score_func : str
        What kind of score is used,
            none: score = $x$
            logsigmoid: score $log(sigmoid(x))
    """
    def __init__(self, device, config, model_path, sfunc='none'):
        assert sfunc in ['none', 'logsigmoid'], 'score function should be none or logsigmoid'

        self.device = 'cpu' if device < 0 else device
        self.config = config
        self.model_path = model_path
        self.sfunc = sfunc
        if sfunc == 'none':
            self.score_func = none
        else:
            self.score_func = logsigmoid

    def load_model(self):
        config = self.config
        model_path = self.model_path
        # for none score func, use 0.
        # for logsigmoid use original gamma to make the score closer to 0.
        gamma=config['gamma'] if self.sfunc == 'logsigmoid' else 0.0
        model = InferModel(device=self.device,
                           model_name=config['model'],
                           hidden_dim=config['emb_size'],
                           double_entity_emb=config['double_ent'],
                           double_relation_emb=config['double_rel'],
                           gamma=gamma)
        dataset = config['dataset']
        model.load_emb(model_path, dataset)
        self.model = model

    def topK(self, head=None, rel=None, tail=None, bcast=None, k=10):
        if head is None:
            head = F.arange(0, self.model.num_entity)
        else:
            head = F.tensor(head, dtype='int64')
        if rel is None:
            rel = F.arange(0, self.model.num_rel)
        else:
            rel = F.tensor(rel, dtype='int64')
        if tail is None:
            tail = F.arange(0, self.model.num_entity)
        else:
            tail = F.tensor(tail, dtype='int64')

        num_head = F.shape(head)[0]
        num_rel = F.shape(rel)[0]
        num_tail = F.shape(tail)[0]

        if bcast is None:
            result = []
            raw_score = self.model.score(head, rel, tail)
            score = self.score_func(raw_score)
            idx = F.arange(0, num_head * num_rel * num_tail)

            sidx = F.argsort(score, dim=0, descending=True)
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
                raw_score = self.model.score(F.unsqueeze(head[i], 0), rel, tail)
                score = self.score_func(raw_score)
                idx = F.arange(0, num_rel * num_tail)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = idx / num_tail
                rel_idx = idx % num_rel

                result.append((np.full((k,), F.asnumpy(head[i])),
                               F.asnumpy(rel[rel_idx]),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif bcast == 'rel':
            result = []
            for i in range(num_rel):
                raw_score = self.model.score(head, F.unsqueeze(rel[i], 0), tail)
                score = self.score_func(raw_score)
                idx = F.arange(0, num_head * num_tail)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = idx / num_tail
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                               np.full((k,), F.asnumpy(rel[i])),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif bcast == 'tail':
            result = []
            for i in range(num_tail):
                raw_score = self.model.score(head, rel, F.unsqueeze(tail[i], 0))
                score = self.score_func(raw_score)
                idx = F.arange(0, num_head * num_rel)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                rel_idx = idx % num_rel
                idx = idx / num_rel
                head_idx = idx % num_head
                result.append((F.asnumpy(head[head_idx]),
                               F.asnumpy(rel[rel_idx]),
                               np.full((k,), F.asnumpy(tail[i])),
                               F.asnumpy(score)))
        else:
            assert False, 'unknow broadcast type {}'.format(bcast)

        return result

class EmbSimInfor():
    """ Calculate simularity of entity/relation embeddings based on pretained KG embeddings

    Parameters
    ---------
    device : int 
        Device to run the inference, -1 for CPU

    emb_file : dict
        Containing embedding information

    sfunc : str
        What kind of score is used,
            cosine: score = $\frac{x \cdot y}{||x||_2||y||_2}$
            l2: score = $||x - y||_2$
            l1: score = $||x - y||_1$
            dot: score = $x \cdot y$
            ext_jaccard: score = $\frac{x \cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \cdot y}$
    """
    def __init__(self, device, emb_file, sfunc='cosine', batch_size=16384):
        self.device = 'cpu' if device < 0 else device
        self.emb_file = emb_file
        self.sfunc = sfunc
        if sfunc == 'cosine':
            self.sim_func = cosine_dist
        elif sfunc == 'l2':
            self.sim_func = l2_dist
        elif sfunc == 'l1':
            self.sim_func = l1_dist
        elif sfunc == 'dot':
            self.sim_func = dot_dist
        elif sfunc == 'ext_jaccard':
            self.sim_func = extended_jaccard_dist
        self.batch_size = batch_size

    def load_emb():
        self.emb = np.load(self.emb_file)
            
    def topK(self, head=None, tail=None, bcast=None, pair_ws=False, k=10):
        if head is None:
            head = F.arange(0, self.emb.shape[0])
        if tail is None:
            tail = F.arange(0, self.emb.shape[0])

        head_emb = self.emb[head]
        tail_emb = self.emb[tail]
        if pair_ws is True:
            result = []
            score = self.sim_func(head_emb, tail_emb, pw=True)
            if self.sfunc == 'cosine' or self.sfunc == 'dot' or self.sfunc == 'ext_jaccard':
                sidx = F.argsort(score, dim=0, descending=True)
            else:
                sidx = F.argsort(score, dim=0, descending=False)

            sidx = sidx[:k]
            score = score[sidx]

            result.append((F.asnumpy(head[sidx]),
                           F.asnumpy(tail[sidx]),
                           F.asnumpy(score)))
        else:
            num_head = head.shape[0]
            num_tail = tail.shape[0]
            batch_size = self.batch_size
            
            # chunked cal score
            score = []
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                            if (i + 1) * batch_size < num_head \
                                            else num_head]
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                    if (j + 1) * batch_size < num_tail \
                                                    else num_tail]
                    s_score.append(self.sim_func(sh_emb, st_emb))
                score.append(F.cat(s_score, dim=1))
            score = F.cat(score, dim=0)

            if bcast is None:
                result = []
                idx = F.arange(0, num_head * num_tail)
                score = F.reshape(score, (num_head * num_tail, ))

                if self.sfunc == 'cosine' or self.sfunc == 'dot' or self.sfunc == 'ext_jaccard':
                    sidx = F.argsort(score, dim=0, descending=True)
                else:
                    sidx = F.argsort(score, dim=0, descending=False)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = idx / num_tail
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                           F.asnumpy(tail[tail_idx]),
                           F.asnumpy(score)))

            elif bcast == 'head':
                result = []
                if self.sfunc == 'cosine' or self.sfunc == 'dot' or self.sfunc == 'ext_jaccard':
                    sidx = F.argsort(score, dim=1, descending=True)
                else:
                    sidx = F.argsort(score, dim=1, descending=False)

                for i in range(num_head):
                    idx = F.arange(0, num_tail)
                    i_idx = sidx[i][:k]
                    i_score = score[i][i_idx]
                    idx = idx[i_idx]

                    result.append((np.full((k,), F.asnumpy(head[i])),
                                  F.asnumpy(tail[idx]),
                                  F.asnumpy(i_score)))
            elif bcast == 'tail':
                result = []
                score = F.swapaxes(score, 0, 1)
                if self.sfunc == 'cosine' or self.sfunc == 'dot' or self.sfunc == 'ext_jaccard':
                    sidx = F.argsort(score, dim=1, descending=True)
                else:
                    sidx = F.argsort(score, dim=1, descending=False)

                for i in range(num_tail):
                    idx = F.arange(0, num_head)
                    # chunked cal score
                    i_idx = sidx[i][:k]
                    i_score = score[i][i_idx]
                    idx = idx[i_idx]

                    result.append((F.asnumpy(head[idx]),
                                   np.full((k,), F.asnumpy(tail[i])),
                                   F.asnumpy(i_score)))
            else:
                assert False, 'unknow broadcast type {}'.format(bcast)

        return result



